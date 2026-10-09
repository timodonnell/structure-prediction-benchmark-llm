"""Scientific scoring contracts: correspondence, invariance and invalid-output accounting."""

from pathlib import Path

import gemmi
import numpy as np
import pytest
import yaml

from pdbthink.prediction.metrics import lddt_ca, score_structure, tm_scores
from pdbthink.prediction.pdb import extract_pdb, format_backbone, sanitise_reference, validate_pdb
from pdbthink.prediction.report import summarise_prediction
from pdbthink.scoring import score_response
from pdbthink.util import REPO_ROOT


@pytest.fixture
def case():
    source = Path(__file__).parent / "fixtures/crambin_processed.pdb"
    structure = gemmi.read_structure(str(source))
    residues = list(structure[0][0])
    sequence = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)
    cfg = yaml.safe_load((REPO_ROOT / "configs/prediction_v1.yaml").read_text())
    parameters = {
        "sequence": sequence,
        "metrics": cfg["metrics"],
        "geometry_diagnostics": cfg["geometry_diagnostics"],
    }
    pdb = sanitise_reference(source.read_text(), sequence, cfg["metrics"]["backbone_atoms"])
    backbone = validate_pdb(pdb, sequence, cfg["metrics"]["backbone_atoms"])
    xyz = np.array(
        [
            [backbone.coordinates[(i, a)] for a in cfg["metrics"]["backbone_atoms"]]
            for i in range(1, len(sequence) + 1)
        ]
    )
    return sequence, xyz, pdb, parameters


def test_fixed_columns_and_final_block(case):
    sequence, _, pdb, parameters = case
    wrapped = "Reasoning text that is ignored.\nFINAL:\n```pdb\n" + pdb + "```\n"
    recovered = extract_pdb(wrapped)
    assert recovered == pdb
    assert validate_pdb(recovered, sequence, parameters["metrics"]["backbone_atoms"]).valid


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda p: p.replace("END\n", ""), "missing_end"),
        (lambda p: p.replace("END\n", "END\n" + p.splitlines()[0] + "\n"), "record_after_end"),
        (lambda p: "\n".join(p.splitlines()[1:]) + "\n", "missing_backbone_atoms"),
        (lambda p: p.splitlines()[0] + "\n" + p, "duplicate_atom"),
        (lambda p: p.replace("ATOM      1", "HETATM    1"), "unsupported_record"),
        (lambda p: p[:30] + "     nan" + p[38:], "invalid_atom_record"),
        (lambda p: p[:21] + "B" + p[22:], "chain_must_be_A"),
        (lambda p: p[:17] + ("GLY" if p[17:20] != "GLY" else "ALA") + p[20:], "inconsistent_residue_name"),
    ],
)
def test_bad_pdb_is_not_scored(case, mutation, reason):
    sequence, _, pdb, parameters = case
    broken = mutation(pdb)
    check = validate_pdb(broken, sequence, parameters["metrics"]["backbone_atoms"])
    assert not check.valid and reason in check.errors
    outcome = score_structure(broken, {"reference_pdb": pdb}, parameters)
    assert outcome["score"] == 0 and not outcome["accuracy_eligible"]
    assert "esmfold" not in outcome


def test_lddt_uses_mean_residue_not_global_pairs():
    reference = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=float)
    predicted = reference.copy()
    predicted[0, 0] = -2
    settings = {
        "lddt_inclusion_radius": 2.5,
        "lddt_min_sequence_separation": 1,
        "lddt_thresholds": [0.5, 1, 2, 4],
    }
    # Neighbour counts 2,2,3,1; fractions .25,.625,.75,1 respectively.
    assert lddt_ca(predicted, reference, settings) == pytest.approx(0.65625)


def test_lddt_rotation_and_mirror_invariance(case):
    _, xyz, _, parameters = case
    ca = xyz[:, 1]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert lddt_ca(ca @ rotation + 50, ca, parameters["metrics"]) == 1
    assert lddt_ca(ca * np.array([-1, 1, 1]), ca, parameters["metrics"]) == 1


@pytest.mark.skipif(
    not (REPO_ROOT / "data/tools/TMscore").exists(),
    reason="install pinned TMscore for native integration tests",
)
def test_official_metrics_rigid_transform_and_reflection(case):
    sequence, xyz, pdb, parameters = case
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = format_backbone(sequence, xyz @ rotation + 30, parameters["metrics"]["backbone_atoms"])
    result = score_response(
        "FINAL\n" + transformed, "pdb_structure", {"reference_pdb": pdb}, parameters=parameters
    )
    assert result["score"]["score"] == 1
    assert result["score"]["esmfold"]["tm_score"] == 1
    assert result["score"]["esmfold"]["gdt_ts"] == 1
    mirror = tm_scores(xyz[:, 1] * [-1, 1, 1], xyz[:, 1], sequence)
    assert mirror["tm_score"] < 0.9
    assert mirror["gdt_ts"] < 1


@pytest.mark.skipif(
    not (REPO_ROOT / "data/tools/TMscore").exists(),
    reason="install pinned TMscore for native integration tests",
)
def test_bad_geometry_is_scored_and_truncation_is_not_eligible(case):
    sequence, xyz, pdb, parameters = case
    distorted = format_backbone(sequence, xyz * 2, parameters["metrics"]["backbone_atoms"])
    result = score_structure(distorted, {"reference_pdb": pdb}, parameters)
    assert result["accuracy_eligible"]
    assert result["geometry"]["ca_chain_breaks"] > 0
    assert result["esmfold"]["lddt_ca"] < 1
    truncated = score_response(
        "FINAL\n" + pdb, "pdb_structure", {"reference_pdb": pdb}, parameters=parameters, truncated=True
    )
    assert truncated["score"]["score"] == 0
    assert not truncated["score"]["accuracy_eligible"]
    assert "esmfold" not in truncated["score"]


def test_summary_keeps_invalid_outputs_in_denominator():
    rows = [
        {
            "semantic_instance_id": str(i),
            "model_id": "mock",
            "detail": {
                "syntax_valid": True,
                "accuracy_eligible": i == 0,
                "esmfold": {"lddt_ca": 1, "tm_score": 1, "gdt_ts": 1, "gdt_ha": 1, "rmsd_ca": 0},
            },
        }
        for i in range(2)
    ]
    stats = summarise_prediction(rows, samples=100)
    metric = stats["accuracy"]["esmfold"]["lddt_ca"]
    assert metric["conditional"]["mean"] == 1
    assert metric["zero_for_invalid"]["mean"] == 0.5
    assert stats["protocol_valid"]["mean"] == 0.5


def test_prediction_scores_cannot_enter_coordinate_macro():
    from pdbthink.evaluation.score import summarise

    common = {
        "model_id": "test",
        "format_error": False,
        "refusal": False,
        "truncated": False,
        "api_error": False,
    }
    coordinate = {
        **common,
        "task_track": "coordinate_reasoning",
        "question_family": "G04",
        "semantic_instance_id": "coordinate",
        "render_id": "coordinate-render",
        "score": 0.75,
    }
    prediction = {
        **common,
        "task_track": "sequence_prediction",
        "question_family": "F01",
        "semantic_instance_id": "prediction",
        "render_id": "prediction-render",
        "score": 0.0,
        "detail": {"syntax_valid": False, "protocol_valid": False, "accuracy_eligible": False},
    }
    summary = summarise([coordinate, prediction])
    assert summary["macro_score"] == 0.75
    assert summary["sequence_prediction"]["n_examples"] == 1
    assert summarise([prediction])["macro_score"] is None


def test_experimental_reference_is_never_labelled_esmfold(case, monkeypatch):
    sequence, _, pdb, parameters = case
    monkeypatch.setattr(
        "pdbthink.prediction.metrics.tm_scores",
        lambda *a: {"tm_score": 1.0, "gdt_ts": 1.0, "gdt_ha": 1.0, "rmsd_ca": 0.0},
    )
    result = score_structure(pdb, {"reference_pdb": pdb, "reference_kind": "experimental_pdb"}, parameters)
    assert result["experimental_pdb"]["lddt_ca"] == 1
    assert "esmfold" not in result
    assert result["score"] == 1


def test_designed_and_experimental_panels_cannot_be_pooled():
    with pytest.raises(ValueError, match="separately"):
        summarise_prediction([{"question_family": "F01"}, {"question_family": "F02"}])


def test_invalid_experimental_prediction_has_zero_accuracy():
    rows = [
        {
            "question_family": "F02",
            "semantic_instance_id": "x",
            "model_id": "m",
            "detail": {"accuracy_eligible": False},
        }
    ]
    stats = summarise_prediction(rows, samples=10)
    assert stats["primary_reference"] == "experimental_pdb"
    assert "esmfold" not in stats["accuracy"]
    assert stats["accuracy"]["experimental_pdb"]["lddt_ca"]["zero_for_invalid"]["mean"] == 0
