"""Protect the published source audit and the standalone input boundary."""

import hashlib
import io
import json
import tarfile
from collections import Counter

import pytest

from pdbthink.prompts.library import prediction_prompt_fingerprint
from pdbthink.util import REPO_ROOT, sha256_file
from scripts.fetch_inputs import install


def archive_case(tmp_path, name="data/cache/structures/example.cif"):
    archive = tmp_path / "inputs.tar.gz"
    data = b"test fixture\n"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    manifest = {"sha256": sha256_file(archive),
                "files": {name: hashlib.sha256(data).hexdigest()}}
    return archive, manifest


def test_archive_round_trip_is_idempotent(tmp_path):
    archive, manifest = archive_case(tmp_path)
    destination = tmp_path / "repo"
    install(archive, destination, manifest)
    install(archive, destination, manifest)
    assert (destination / next(iter(manifest["files"]))).read_bytes() == b"test fixture\n"


def test_archive_hash_is_checked_before_install(tmp_path):
    archive, manifest = archive_case(tmp_path)
    manifest["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive hash"):
        install(archive, tmp_path / "repo", manifest)
    assert not (tmp_path / "repo").exists()


def test_archive_refuses_to_overwrite_changed_inputs(tmp_path):
    archive, manifest = archive_case(tmp_path)
    destination = tmp_path / "repo"
    install(archive, destination, manifest)
    (destination / next(iter(manifest["files"]))).write_text("local changes")
    with pytest.raises(ValueError, match="overwrite"):
        install(archive, destination, manifest)


@pytest.mark.parametrize("name", ["../escaped", "/tmp/escaped"])
def test_archive_rejects_unsafe_paths(tmp_path, name):
    archive, manifest = archive_case(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe"):
        install(archive, tmp_path / "repo", manifest)


def test_frozen_query_and_screen_counts():
    root = REPO_ROOT / "data/prediction_sources/pdb_v1"
    response = json.loads((root / "response.json").read_text())
    sources = json.loads((root / "sources.json").read_text())
    excluded = set(json.loads((REPO_ROOT / "data/provenance/coordinate_exclusions.json").read_text()))
    representatives = [r["identifier"] for r in response["result_set"]]
    assert len(representatives) == 1665
    assert sum(e.split("_")[0] in excluded for e in representatives) == 14
    assert set(sources["entities"]) == {e for e in representatives if e.split("_")[0] not in excluded}
    assert len(sources["entities"]) == 1651
    rows = [json.loads(line) for line in
            (REPO_ROOT / "data/frozen/prediction_pdb_v1/rejections.jsonl").read_text().splitlines()]
    counts = Counter(row["reason"] for row in rows if "reason" in row)
    assert sum(counts.values()) == 1627
    selection = json.loads((REPO_ROOT / "data/frozen/prediction_pdb_v1/selection.json").read_text())
    assert selection["n_eligible"] == 24
    assert len(selection["selection"]) == 10


def test_original_snapshot_and_manifest_hashes_are_retained():
    configs = [("prediction_v1_panel.json", "prediction_pilot_v1"),
               ("prediction_pdb_v1_panel.json", "prediction_pdb_v1")]
    for config, panel in configs:
        expected = json.loads((REPO_ROOT / "configs" / config).read_text())
        for name, key in [("instances.jsonl", "instance_manifest_sha256"),
                          ("renders.jsonl", "render_manifest_sha256")]:
            assert sha256_file(REPO_ROOT / "data/frozen" / panel / name) == expected[key]
        for name, digest in expected.get("source_snapshots", {}).items():
            assert sha256_file(REPO_ROOT / "data/prediction_sources/pdb_v1" / name) == digest
    assert prediction_prompt_fingerprint() == "da6f56ab5c1fc828"
