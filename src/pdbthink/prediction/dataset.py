"""Frozen, sequence-only pilot selected from the completed MarinFold exp278 screen."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from ..dataset import BuildResult, write_dataset
from ..prompts.library import (
    PREDICTION_PROMPT_VERSION,
    PREDICTION_QUESTION,
    PREDICTION_SYSTEM_PROMPT,
    prediction_prompt_fingerprint,
)
from ..representations.tokens import count_tokens
from ..schemas import RenderedVariant, SemanticInstance
from ..util import gold_hash, sha256_bytes, sha256_text, stable_hash, write_json
from .metrics import lddt_ca, tm_scores
from .pdb import sanitise_reference, validate_pdb


def _screen_rows(path) -> dict:
    rows = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if row["stem"] in rows:
                raise ValueError(f"duplicate screen record: {row['stem']}")
            rows[row["stem"]] = row
    return rows


def sequence_similarity(first: str, second: str) -> tuple[float, float]:
    """Global affine-gap alignment identity and paired coverage / shorter length."""
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner(
        mode="global", match_score=2, mismatch_score=-1, open_gap_score=-10, extend_gap_score=-0.5
    )
    alignment = aligner.align(first, second)[0]
    pairs = alignment.indices
    present = (pairs[0] >= 0) & (pairs[1] >= 0)
    identical = sum(first[i] == second[j] for i, j in pairs[:, present].T)
    denominator = min(len(first), len(second))
    return identical / denominator, int(present.sum()) / denominator


def structural_similarity(first: dict, second: dict) -> float:
    from tmtools import tm_align

    alignment = tm_align(first["ca"], second["ca"], first["sequence"], second["sequence"])
    return max(alignment.tm_norm_chain1, alignment.tm_norm_chain2)


def build_prediction_dataset(config_path, candidates, retention_path, quality_path, output):
    import pyarrow.parquet as pq

    config_path, output = Path(config_path), Path(output)
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)
    if sum(config["length_quotas"].values()) != config["count"]:
        raise ValueError("length quotas must sum to count")
    retention = _screen_rows(retention_path)
    quality = _screen_rows(quality_path)
    source_hashes = {
        Path(p).name + ":" + Path(p).parent.name: sha256_bytes(Path(p).read_bytes())
        for p in sorted(map(Path, candidates))
    }
    config_hash = sha256_text(config_text)
    pool, excluded = [], []
    seen = set()
    for path in sorted(map(Path, candidates)):
        for row in pq.read_table(path).to_pylist():
            stem = row["stem"]
            if stem in seen:
                raise ValueError(f"duplicate source candidate: {stem}")
            seen.add(stem)
            screen = retention.get(stem, {})
            q = quality.get(stem, {})
            if (
                screen.get("selected") != "True"
                or screen.get("sequence_excluded") != "False"
                or screen.get("structure_excluded") != "False"
            ):
                continue
            if not screen.get("fine_cluster") or not q:
                raise ValueError(f"incomplete upstream screen: {stem}")
            if (
                not np.isfinite([float(q["plddt"]), float(q["scrmsd"])]).all()
                or q["quality_pass"] != "True"
                or not row["quality_pass"]
                or not row["ca_geometry_pass"]
                or float(q["plddt"]) < config["selection"]["min_plddt"]
                or float(q["scrmsd"]) > config["selection"]["max_self_consistency_rmsd"]
            ):
                continue
            sequence = row["sequence"]
            try:
                pdb = sanitise_reference(row["pdb_content"], sequence, config["metrics"]["backbone_atoms"])
                backbone = validate_pdb(pdb, sequence, config["metrics"]["backbone_atoms"])
                if not backbone.valid:
                    raise ValueError(";".join(backbone.errors))
                original = np.asarray(row["original_ca"], dtype=float)
                if original.shape != (len(sequence), 3) or not np.isfinite(original).all():
                    raise ValueError("invalid original backbone")
            except (ValueError, KeyError, RuntimeError) as exc:
                excluded.append({"stem": stem, "reason": str(exc)})
                continue
            ca = backbone.ca
            singular = np.linalg.svd(ca - ca.mean(axis=0), compute_uv=False)
            if singular[0] == 0:
                raise ValueError(f"degenerate reference: {stem}")
            features = [
                float(q["refolded_fraction_alpha"]),
                float(q["refolded_fraction_beta"]),
                float(q["refolded_fraction_coil"]),
                singular[1] / singular[0],
                singular[2] / singular[0],
            ]
            if not np.isfinite(features).all() or not np.isclose(sum(features[:3]), 1):
                raise ValueError(f"invalid diversity measurements: {stem}")
            pool.append(
                {
                    **row,
                    "reference_pdb": pdb,
                    "ca": ca,
                    "original": original,
                    "quality": q,
                    "screen": screen,
                    "features": np.array(features),
                    "source_hash": sha256_text(row["pdb_content"]),
                    "tie": stable_hash([config["seed"], stem]),
                }
            )
    selected, selection_log = [], []
    # Long examples have fewer quality-passing candidates. Fill scarce strata first.
    quotas = {int(k): int(v) for k, v in config["length_quotas"].items()}
    for length in sorted(quotas, reverse=True):
        for _ in range(quotas[length]):
            eligible = [
                r
                for r in pool
                if len(r["sequence"]) == length
                and r["stem"] not in {s["stem"] for s in selected}
                and r["screen"]["fine_cluster"] not in {s["screen"]["fine_cluster"] for s in selected}
            ]

            def distance(row):
                return (
                    min(np.linalg.norm(row["features"] - s["features"]) for s in selected)
                    if selected
                    else 0.0
                )

            eligible.sort(key=lambda r: (-distance(r), r["tie"]))
            chosen = None
            for row in eligible:
                comparisons = []
                for other in selected:
                    tm = structural_similarity(row, other)
                    identity, coverage = sequence_similarity(row["sequence"], other["sequence"])
                    comparisons.append(
                        {"other": other["stem"], "tm": tm, "identity": identity, "coverage": coverage}
                    )
                    if tm > config["selection"]["max_pairwise_tm_similarity"] or (
                        identity > config["selection"]["max_sequence_identity"]
                        and coverage >= config["selection"]["sequence_coverage"]
                    ):
                        break
                else:
                    chosen = row
                    selection_log.append({"stem": row["stem"], "comparisons": comparisons})
                    break
            if chosen is None:
                raise ValueError(
                    f"cannot fill length {length} quota without relaxing diversity; no dataset written"
                )
            selected.append(chosen)
    selected.sort(key=lambda r: (len(r["sequence"]), r["tie"]))
    result = BuildResult()
    selection_rows = []
    for index, row in enumerate(selected, 1):
        sequence = row["sequence"]
        identifier = f"F01-{stable_hash([config_hash, row['stem'], row['source_hash']])[:16]}"
        group = f"synthetic-{index:02d}"
        gold = {"reference_pdb": row["reference_pdb"], "proteina_ca": row["original"].tolist()}
        parameters = {
            "sequence": sequence,
            "metrics": config["metrics"],
            "geometry_diagnostics": config["geometry_diagnostics"],
        }
        instance = SemanticInstance(
            semantic_instance_id=identifier,
            task_track="sequence_prediction",
            question_family="F01",
            question_version=PREDICTION_PROMPT_VERSION,
            protein_group_id=group,
            source_type="synthetic",
            source_entries=[row["stem"]],
            source_file_sha256s=[row["source_hash"]],
            question_parameters=parameters,
            answer_schema="pdb_structure",
            gold_answer=gold,
            gold_evidence={
                "cluster": group,
                "source": config["source"],
                "source_prefix": row["source_prefix"],
                "quality": row["quality"],
                "upstream_screen": row["screen"],
                "config_sha256": config_hash,
                "prompt_fingerprint": prediction_prompt_fingerprint(),
                "sequence_sha256": sha256_text(sequence),
            },
            selection_margins={
                "plddt_above_min": float(row["quality"]["plddt"]) - config["selection"]["min_plddt"],
                "rmsd_below_max": config["selection"]["max_self_consistency_rmsd"]
                - float(row["quality"]["scrmsd"]),
            },
            definition_version=config["definition_version"],
            criteria_passed=["upstream_reference_screen", "backbone_complete", "pairwise_diversity"],
        )
        prompt = PREDICTION_QUESTION.format(sequence=sequence, length=len(sequence))
        render = RenderedVariant(
            render_id=f"{identifier}::sequence_only",
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
            question_family="F01",
            protein_group_id=group,
            answer_schema="pdb_structure",
            gold_answer=gold,
            gold_sha256=gold_hash(gold),
            input_token_count=count_tokens(PREDICTION_SYSTEM_PROMPT + "\n" + prompt, "cl100k_base")[0],
            tokenizer="cl100k_base",
        )
        result.instances.append(instance)
        result.renders.append(render)
        selection_rows.append(
            {
                "example": group,
                "semantic_instance_id": identifier,
                "stem": row["stem"],
                "length": len(sequence),
                "plddt": float(row["quality"]["plddt"]),
                "scrmsd": float(row["quality"]["scrmsd"]),
                "alpha": row["features"][0],
                "beta": row["features"][1],
                "coil": row["features"][2],
                "fine_cluster": row["screen"]["fine_cluster"],
                "reference_to_design": {
                    "lddt_ca": lddt_ca(row["ca"], row["original"], config["metrics"]),
                    **tm_scores(row["ca"], row["original"], sequence),
                },
            }
        )
    pairwise = np.eye(len(selected))
    for i, row in enumerate(selected):
        for j in range(i):
            pairwise[i, j] = pairwise[j, i] = structural_similarity(row, selected[j])
    result.stats = {
        "name": "pdbthink_prediction_pilot",
        "version": config["dataset_version"],
        "task_track": "sequence_prediction",
        "definition_version": config["definition_version"],
        "config_sha256": config_hash,
        "prompt_fingerprint": prediction_prompt_fingerprint(),
        "source": config["source"],
        "source_parquet_sha256s": source_hashes,
        "retention_sha256": sha256_bytes(Path(retention_path).read_bytes()),
        "quality_sha256": sha256_bytes(Path(quality_path).read_bytes()),
        "n_source": len(seen),
        "n_eligible": len(pool),
        "n_instances": len(selected),
        "n_renders": len(selected),
        "realised_counts": {"F01": len(selected)},
        "selection": selection_rows,
        "pairwise_tm_similarity": pairwise.tolist(),
        "selection_log": selection_log,
        "excluded_reference_records": excluded,
    }
    write_dataset(result, output)
    write_json(output / "selection.json", result.stats)
    return result.stats


def validate_prediction_render(instance, render) -> list[str]:
    errors = []
    sequence = instance.question_parameters["sequence"]
    if instance.question_family == "F02":
        if instance.source_type != "pdb" or not instance.release_dates:
            errors.append("experimental prediction reference lacks PDB provenance")
        if instance.gold_answer.get("reference_kind") != "experimental_pdb":
            errors.append("F02 requires an experimental PDB reference")
        if render.gold_answer.get("reference_kind") != "experimental_pdb":
            errors.append("F02 render reference kind mismatch")
        reference = validate_pdb(
            instance.gold_answer["reference_pdb"],
            sequence,
            instance.question_parameters["metrics"]["backbone_atoms"],
        )
        if not reference.valid:
            errors.append("experimental prediction reference backbone is invalid")
    expected = PREDICTION_QUESTION.format(sequence=sequence, length=len(sequence))
    if render.user_prompt != expected or render.system_prompt != PREDICTION_SYSTEM_PROMPT:
        errors.append("sequence prediction prompt does not match versioned template")
    if (
        render.prompt_version != PREDICTION_PROMPT_VERSION
        or instance.question_version != PREDICTION_PROMPT_VERSION
    ):
        errors.append("prediction prompt version mismatch")
    if render.task_track != "sequence_prediction" or render.representation != "sequence_only":
        errors.append("prediction track/representation mismatch")
    if render.atom_count != 0 or render.displayed_coordinates_sha256 != sha256_text(""):
        errors.append("sequence-only prompt claims displayed coordinates")
    if instance.gold_evidence.get("sequence_sha256") != sha256_text(sequence):
        errors.append("sequence hash mismatch")
    if instance.gold_evidence.get("prompt_fingerprint") != prediction_prompt_fingerprint():
        errors.append("prediction prompt fingerprint mismatch")
    return errors
