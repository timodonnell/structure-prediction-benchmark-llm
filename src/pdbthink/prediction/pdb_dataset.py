"""Experimental sequence-prediction panels require complete, unambiguous monomers."""

from __future__ import annotations

import json
from pathlib import Path

import gemmi
import numpy as np
import yaml

from ..acquisition.cache import StructureCache
from ..dataset import BuildResult, write_dataset
from ..prompts.library import (
    PREDICTION_PROMPT_VERSION,
    PREDICTION_QUESTION,
    PREDICTION_SYSTEM_PROMPT,
    prediction_prompt_fingerprint,
)
from ..representations.tokens import count_tokens
from ..schemas import RejectionRecord, RenderedVariant, SemanticInstance
from ..util import REPO_ROOT, gold_hash, sha256_bytes, sha256_text, stable_hash, write_json, write_jsonl
from .dataset import sequence_similarity, structural_similarity
from .pdb import format_backbone, validate_pdb

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


class SourceIntegrityError(Exception):
    """A changed source is a versioning failure, not a candidate rejection."""


def experimental_backbone(path, entity_id, settings, geometry_settings):
    """Never close gaps by dropping unobserved residues from the supplied sequence."""
    block = gemmi.cif.read_file(str(path)).sole_block()
    structure = gemmi.make_structure_from_block(block)
    if len(structure) != 1:
        raise ValueError("multiple_coordinate_models")
    polymers = list(
        block.find(
            ["_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.pdbx_seq_one_letter_code_can"]
        )
    )
    if (
        len(polymers) != 1
        or polymers[0][0] != entity_id
        or gemmi.cif.as_string(polymers[0][1]) != "polypeptide(L)"
    ):
        raise ValueError("not_one_protein_entity")
    sequence = "".join(gemmi.cif.as_string(polymers[0][2]).split())
    if not sequence or set(sequence) - STANDARD:
        raise ValueError("nonstandard_entity_sequence")
    counts = list(block.find_values("_pdbx_struct_assembly.oligomeric_count"))
    if not counts or any(value != "1" for value in counts):
        raise ValueError("assembly_not_unambiguously_monomeric")
    sources = {r[0]: r[1] for r in block.find(["_entity.id", "_entity.src_method"])}
    if sources.get(entity_id) == "syn":
        raise ValueError("synthetic_polymer_source")
    chains = [(chain, list(chain.get_polymer())) for chain in structure[0]]
    chains = [(chain, residues) for chain, residues in chains if residues]
    if len(chains) != 1:
        raise ValueError("not_one_observed_protein_chain")
    chain, residues = chains[0]
    if any(
        r.entity_type != gemmi.EntityType.Polymer and r.name not in ("HOH", "DOD", "WAT")
        for c in structure[0]
        for r in c
    ):
        raise ValueError("nonwater_ligand_present")
    actual = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)
    if actual != sequence or [r.label_seq for r in residues] != list(range(1, len(sequence) + 1)):
        raise ValueError("unobserved_or_mismatched_entity_residues")
    coordinates = []
    for residue in residues:
        for name in settings["backbone_atoms"]:
            atoms = [a for a in residue if a.name == name]
            if len(atoms) != 1 or atoms[0].altloc != "\x00" or not np.isclose(atoms[0].occ, 1):
                raise ValueError("missing_or_ambiguous_backbone_atom")
            coordinates.append(list(atoms[0].pos))
    pdb = format_backbone(sequence, np.asarray(coordinates), settings["backbone_atoms"])
    parsed = validate_pdb(pdb, sequence, settings["backbone_atoms"])
    if not parsed.valid:
        raise ValueError("invalid_canonical_reference")
    lengths = np.linalg.norm(np.diff(parsed.ca, axis=0), axis=1)
    if np.any(
        (lengths < geometry_settings["adjacent_ca_min"]) | (lengths > geometry_settings["adjacent_ca_max"])
    ):
        raise ValueError("reference_chain_break")
    return {
        "sequence": sequence,
        "reference_pdb": pdb,
        "ca": parsed.ca,
        "chain": chain.name,
        "author_residue_ids": [str(r.seqid) for r in residues],
    }


def build_pdb_prediction_dataset(config_path, source_manifest, output):
    config_text = Path(config_path).read_text()
    config = yaml.safe_load(config_text)
    metric_config = yaml.safe_load((REPO_ROOT / config["metric_config"]).read_text())
    config_hash = stable_hash(config_text, metric_config)
    manifest = json.loads(Path(source_manifest).read_text())
    cache = StructureCache(offline=True)
    pool, rejections = [], []
    for entity in sorted(manifest["entities"]):
        entry, entity_id = entity.split("_")
        try:
            record = cache.get_pdb(entry)
            if sha256_bytes(Path(record.path).read_bytes()) != manifest["source_hashes"][entry]:
                raise SourceIntegrityError(f"source_hash_mismatch: {entry}")
            if (
                not record.release_date
                or not config["selection"]["release_from"]
                <= record.release_date
                <= config["selection"]["release_to"]
            ):
                raise ValueError("release_date_outside_window")
            if (
                record.experimental_method != config["selection"]["method"]
                or record.resolution is None
                or record.resolution > config["selection"]["max_resolution"]
            ):
                raise ValueError("experimental_quality_filter")
            if any(
                word in (record.title or "").lower()
                for word in ["de novo", "de-novo", "designed", "artificial"]
            ):
                raise ValueError("design_related_title")
            row = experimental_backbone(
                record.path,
                entity_id,
                metric_config["metrics"],
                metric_config["geometry_diagnostics"],
            )
            s = np.linalg.svd(row["ca"] - row["ca"].mean(axis=0), compute_uv=False)
            row.update(
                entity=entity,
                record=record,
                features=np.r_[
                    s[1:] / s[0], len(row["sequence"]) / max(x["max"] for x in config["length_strata"])
                ],
                tie=stable_hash(config["seed"], entity),
            )
            pool.append(row)
        except (ValueError, KeyError, RuntimeError) as exc:
            rejections.append(
                RejectionRecord(
                    candidate_id=entity,
                    question_family="F02",
                    protein_group_id=entity,
                    reason=str(exc),
                    criteria_failed=[str(exc)],
                ).model_dump()
            )
    selected = []
    for stratum in sorted(config["length_strata"], key=lambda s: s["min"], reverse=True):
        for _ in range(stratum["count"]):
            candidates = [
                r
                for r in pool
                if stratum["min"] <= len(r["sequence"]) <= stratum["max"]
                and r["entity"] not in {s["entity"] for s in selected}
            ]
            candidates.sort(
                key=lambda r: (
                    -min((np.linalg.norm(r["features"] - s["features"]) for s in selected), default=0),
                    r["tie"],
                )
            )
            for candidate in candidates:
                for other in selected:
                    identity, coverage = sequence_similarity(candidate["sequence"], other["sequence"])
                    shared = set(candidate["record"].publications) & set(other["record"].publications)
                    if (
                        shared
                        or structural_similarity(candidate, other)
                        > config["selection"]["max_pairwise_tm_similarity"]
                        or (
                            identity > config["selection"]["max_sequence_identity"]
                            and coverage >= config["selection"]["sequence_coverage"]
                        )
                    ):
                        break
                else:
                    selected.append(candidate)
                    break
            else:
                write_jsonl(Path(output) / "rejections.jsonl", rejections)
                raise ValueError(f"cannot fill frozen length stratum {stratum}; eligible pool {len(pool)}")
    if len(selected) != config["count"]:
        raise ValueError("panel_count_mismatch")
    result = BuildResult()
    selection = []
    for n, row in enumerate(sorted(selected, key=lambda r: (len(r["sequence"]), r["tie"])), 1):
        record = row["record"]
        sequence = row["sequence"]
        group = f"experimental-{n:02d}"
        identifier = "F02-" + stable_hash(config_hash, row["entity"], record.sha256)[:16]
        gold = {"reference_pdb": row["reference_pdb"], "reference_kind": "experimental_pdb"}
        parameters = {
            "sequence": sequence,
            "metrics": metric_config["metrics"],
            "geometry_diagnostics": metric_config["geometry_diagnostics"],
        }
        instance = SemanticInstance(
            semantic_instance_id=identifier,
            task_track="sequence_prediction",
            question_family="F02",
            question_version=PREDICTION_PROMPT_VERSION,
            protein_group_id=group,
            source_type="pdb",
            source_entries=[record.entry],
            source_file_sha256s=[record.sha256],
            release_dates=[record.release_date],
            source_publications=record.publications,
            biological_assembly_ids=record.assembly_ids,
            selected_chains=[row["chain"]],
            question_parameters=parameters,
            answer_schema="pdb_structure",
            gold_answer=gold,
            gold_evidence={
                "cluster": row["entity"],
                "reference_kind": "experimental_pdb",
                "config_sha256": config_hash,
                "prompt_fingerprint": prediction_prompt_fingerprint(),
                "sequence_sha256": sha256_text(sequence),
                "entity": row["entity"],
                "author_residue_ids": row["author_residue_ids"],
            },
            definition_version=config["definition_version"],
            selection_margins={
                "resolution_below_max": config["selection"]["max_resolution"] - record.resolution,
                "observed_sequence_coverage": 1.0,
            },
            experimental_method=record.experimental_method,
            resolution=record.resolution,
            criteria_passed=[
                "complete_entity_sequence",
                "monomeric_assembly",
                "unambiguous_backbone",
                "no_nonwater_ligands",
                "pairwise_diversity",
            ],
        )
        prompt = PREDICTION_QUESTION.format(sequence=sequence, length=len(sequence))
        render = RenderedVariant(
            render_id=identifier + "::sequence_only",
            semantic_instance_id=identifier,
            task_track="sequence_prediction",
            representation="sequence_only",
            rotation_seed=0,
            prompt_version=PREDICTION_PROMPT_VERSION,
            system_prompt=PREDICTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            rotation_matrix=np.eye(3).tolist(),
            translation_vector=[0, 0, 0],
            displayed_coordinates_sha256=sha256_text(""),
            question_family="F02",
            protein_group_id=group,
            answer_schema="pdb_structure",
            gold_answer=gold,
            gold_sha256=gold_hash(gold),
            input_token_count=count_tokens(PREDICTION_SYSTEM_PROMPT + "\n" + prompt, "cl100k_base")[0],
            tokenizer="cl100k_base",
        )
        result.instances.append(instance)
        result.renders.append(render)
        selection.append(
            {
                "example": group,
                "semantic_instance_id": identifier,
                "entity": row["entity"],
                "chain": row["chain"],
                "length": len(sequence),
                "release_date": record.release_date,
                "resolution": record.resolution,
                "source_sha256": record.sha256,
            }
        )
    result.stats = {
        "n_instances": len(selected),
        "n_renders": len(selected),
        "realised_counts": {"F02": len(selected)},
        "definition_version": config["definition_version"],
        "config_sha256": config_hash,
        "n_screened": len(manifest["entities"]),
        "n_eligible": len(pool),
        "selection": selection,
    }
    result.rejections = [RejectionRecord(**row) for row in rejections]
    write_dataset(result, output)
    write_json(Path(output) / "selection.json", result.stats)
    return result.stats
