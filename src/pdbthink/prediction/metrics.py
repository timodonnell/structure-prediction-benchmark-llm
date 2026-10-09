"""Fixed-correspondence structural agreement; no sequence realignment at scoring."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..util import REPO_ROOT
from .pdb import Backbone, format_backbone, validate_pdb

TMSCORE_SOURCE_SHA256 = "02d24a35857b1076a432e97c4d63cad7993ac2cc6585a1b85a7c33031c84f8b1"


@lru_cache(maxsize=8)
def _verify_backend(executable: str, modified_ns: int) -> None:
    path = Path(executable)
    metadata_path = path.with_name(path.name + ".build.json")
    if not metadata_path.exists():
        raise FileNotFoundError("TMscore build provenance is missing: run scripts/install_tmscore.py")
    metadata = json.loads(metadata_path.read_text())
    if metadata["source_sha256"] != TMSCORE_SOURCE_SHA256 or (
        metadata["executable_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("TMscore executable does not match its pinned build provenance")


def lddt_ca(predicted: np.ndarray, reference: np.ndarray, settings: dict) -> float:
    """Mean per-residue Cα lDDT, excluding residues with no eligible neighbours."""
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != reference.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("lDDT requires corresponding (L, 3) arrays")
    if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
        raise ValueError("non-finite coordinates")
    distances = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
    delta = np.abs(distances - np.linalg.norm(predicted[:, None] - predicted[None, :], axis=-1))
    indices = np.arange(len(reference))
    eligible = (distances < settings["lddt_inclusion_radius"]) & (
        np.abs(indices[:, None] - indices[None, :]) >= settings["lddt_min_sequence_separation"]
    )
    counts = eligible.sum(axis=1)
    if not np.any(counts):
        raise ValueError("reference has no lDDT neighbourhoods")
    agreement = np.mean([delta < t for t in settings["lddt_thresholds"]], axis=0)
    return float(np.mean((agreement * eligible).sum(axis=1)[counts > 0] / counts[counts > 0]))


def tm_scores(predicted: np.ndarray, reference: np.ndarray, sequence: str) -> dict[str, float]:
    """Official TMscore optimises TM/GDT superpositions with residue-index pairing.

    PDB files are canonicalised to matching indices; TM-align's ability to remap
    residues is intentionally unavailable here. Reflections are never allowed.
    """
    executable = Path(os.environ.get("PDBTHINK_TMSCORE", REPO_ROOT / "data/tools/TMscore"))
    if not executable.is_file():
        raise FileNotFoundError("TMscore is required: run scripts/install_tmscore.py first")
    _verify_backend(str(executable.resolve()), executable.stat().st_mtime_ns)
    with tempfile.TemporaryDirectory(prefix="pdbthink-tm-") as directory:
        paths = [Path(directory) / name for name in ("prediction.pdb", "reference.pdb")]
        for path, xyz in zip(paths, (predicted, reference)):
            path.write_text(format_backbone(sequence, np.asarray(xyz)[:, None, :], ("CA",)))
        result = subprocess.run(
            [str(executable.resolve()), *map(str, paths)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    patterns = {
        "tm_score": r"^TM-score\s*=\s*([\d.]+)",
        "gdt_ts": r"^GDT-TS-score\s*=\s*([\d.]+)",
        "gdt_ha": r"^GDT-HA-score\s*=\s*([\d.]+)",
        "rmsd_ca": r"RMSD of\s+the common residues\s*=\s*([\d.]+)",
    }
    scores = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, result.stdout, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"TMscore did not return {name}: {result.stdout[:1500]}")
        scores[name] = float(match.group(1))
    if any(not 0 <= scores[k] <= 1 for k in ("tm_score", "gdt_ts", "gdt_ha")):
        raise RuntimeError("TMscore returned an out-of-range score")
    return scores


def geometry_diagnostics(backbone: Backbone, settings: dict) -> dict[str, Any]:
    """Report plausibility separately: invalid geometry must still receive accuracy scores."""
    ca = backbone.ca
    steps = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    distances = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    ii, jj = np.triu_indices(len(ca), k=settings["nonlocal_min_sequence_separation"])
    bad_bonds = {}
    for bond, (low, high) in settings["covalent_bonds"].items():
        first, second = bond.split("-")
        pairs = [(i, i + 1 if bond == "C-N" else i) for i in sorted(backbone.residues)]
        lengths = [
            np.linalg.norm(backbone.coordinates[(i, first)] - backbone.coordinates[(j, second)])
            for i, j in pairs
            if (i, first) in backbone.coordinates and (j, second) in backbone.coordinates
        ]
        bad_bonds[bond] = sum(not low <= length <= high for length in lengths)
    return {
        "ca_chain_breaks": int(
            np.sum((steps < settings["adjacent_ca_min"]) | (steps > settings["adjacent_ca_max"]))
        ),
        "nonlocal_ca_clashes": int(np.sum(distances[ii, jj] < settings["nonlocal_ca_clash"])),
        "covalent_bond_outliers": bad_bonds,
    }


def score_structure(text: str, gold: dict, parameters: dict) -> dict[str, Any]:
    settings = parameters["metrics"]
    reference_kind = gold.get("reference_kind", "esmfold")
    if reference_kind not in ("esmfold", "experimental_pdb"):
        raise ValueError("unknown prediction reference kind")
    if settings["tm_source_sha256"] != TMSCORE_SOURCE_SHA256:
        raise ValueError("unsupported TMscore definition; install the matching versioned scorer")
    if len(text) > settings["max_response_characters"]:
        return {
            "score": 0.0,
            "correct": False,
            "syntax_valid": False,
            "protocol_valid": False,
            "accuracy_eligible": False,
            "errors": ["response_too_long"],
        }
    sequence = parameters["sequence"]
    backbone = validate_pdb(text, sequence, settings["backbone_atoms"])
    out: dict[str, Any] = {
        "score": 0.0,
        "correct": False,
        "syntax_valid": backbone.syntax_valid,
        "protocol_valid": backbone.valid,
        "accuracy_eligible": backbone.valid,
        "errors": backbone.errors,
        "atom_records": backbone.atom_records,
        "valid_atom_records": backbone.valid_atom_records,
    }
    if not backbone.valid:
        return out
    reference = validate_pdb(gold["reference_pdb"], sequence, settings["backbone_atoms"])
    if not reference.valid:
        raise ValueError("invalid withheld reference PDB")
    out["geometry"] = geometry_diagnostics(backbone, parameters["geometry_diagnostics"])
    out[reference_kind] = {
        "lddt_ca": lddt_ca(backbone.ca, reference.ca, settings),
        **tm_scores(backbone.ca, reference.ca, sequence),
    }
    if "proteina_ca" in gold:
        target = np.asarray(gold["proteina_ca"], dtype=float)
        out["proteina"] = {
            "lddt_ca": lddt_ca(backbone.ca, target, settings),
            **tm_scores(backbone.ca, target, sequence),
        }
    out["score"] = out[reference_kind]["lddt_ca"]
    # This flag means perfect reference agreement, not evidence of a natural fold.
    out["correct"] = out["score"] == 1.0 and out[reference_kind]["tm_score"] == 1.0
    return out
