"""Protect the query-only ablation from accidentally importing homologue information."""

import pytest
import yaml

from pdbthink.util import REPO_ROOT
from scripts.run_af2_comparator import audit_single_sequence


def test_single_sequence_keeps_other_predictor_settings_fixed():
    msa = yaml.safe_load((REPO_ROOT / "configs/alphafold2_comparator_v1.yaml").read_text())
    single = yaml.safe_load((REPO_ROOT / "configs/alphafold2_single_sequence_v1.yaml").read_text())
    assert single["msa_mode"] == "single_sequence"
    assert single["model_id"] != msa["model_id"]
    assert single["comparator_version"] != msa["comparator_version"]
    for key in ["msa_mode", "note", "model_id", "comparator_version"]:
        msa.pop(key)
        single.pop(key)
    assert single == msa


def test_query_only_alignment_audit(tmp_path):
    path = tmp_path / "query.a3m"
    path.write_text("#4\t1\n>101\nACDE\n>101\nACDE\n")
    assert audit_single_sequence(path, "ACDE")["homologue_rows"] == 0
    assert audit_single_sequence(path, "ACDE")["query_rows"] == 2


@pytest.mark.parametrize("extra", [">homologue\nACDF\n", ">gapped\nAC-E\n", ">empty\n"])
def test_reject_non_query_alignment(tmp_path, extra):
    path = tmp_path / "query.a3m"
    path.write_text(">101\nACDE\n" + extra)
    with pytest.raises(ValueError, match="non-query"):
        audit_single_sequence(path, "ACDE")
