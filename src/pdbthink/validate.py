"""Validate sequence-only panels without importing coordinate-reasoning tasks."""

from pathlib import Path

from .dataset import load_dataset
from .evaluation.runner import dataset_fingerprint
from .prediction.dataset import validate_prediction_render
from .prediction.pdb import validate_pdb
from .util import REPO_ROOT, gold_hash, read_json, sha256_file

PANELS = {
    "F01": ("prediction_pilot_v1", "prediction_v1_panel.json"),
    "F02": ("prediction_pdb_v1", "prediction_pdb_v1_panel.json"),
}


def validate_dataset(directory: str | Path, *, frozen: bool = True) -> dict:
    instances, renders = load_dataset(directory)
    families = {i.question_family for i in instances}
    if len(families) != 1 or not families <= PANELS.keys():
        raise ValueError("Require one nonempty F01 or F02 panel; never pool the panels")
    panel = next(iter(families))
    by_id = {i.semantic_instance_id: i for i in instances}
    if len(by_id) != len(instances) or len({r.render_id for r in renders}) != len(renders):
        raise ValueError("Duplicate instance or render")
    if {r.semantic_instance_id for r in renders} != set(by_id):
        raise ValueError("Instances and renders do not correspond")
    for render in renders:
        instance = by_id[render.semantic_instance_id]
        errors = validate_prediction_render(instance, render)
        for record in (instance, render):
            if record.gold_sha256 != gold_hash(record.gold_answer):
                errors.append("gold hash mismatch")
        if not validate_pdb(
            instance.gold_answer["reference_pdb"],
            instance.question_parameters["sequence"],
            instance.question_parameters["metrics"]["backbone_atoms"],
        ).valid:
            errors.append("invalid reference backbone")
        if errors:
            raise ValueError(f"{render.render_id}: {errors}")
    fingerprint = dataset_fingerprint(renders)
    if frozen:
        expected = read_json(REPO_ROOT / "configs" / PANELS[panel][1])
        if sorted(by_id) != sorted(expected["semantic_instance_ids"]):
            raise ValueError("Frozen panel membership changed")
        if fingerprint != expected["dataset_fingerprint"]:
            raise ValueError("Frozen prompts or reference labels changed")
        for file, key in (("instances.jsonl", "instance_manifest_sha256"),
                          ("renders.jsonl", "render_manifest_sha256")):
            if sha256_file(Path(directory) / file) != expected[key]:
                raise ValueError(f"Frozen {file} changed")
    return {"panel": panel, "instances": len(instances), "renders": len(renders),
            "dataset_fingerprint": fingerprint, "frozen": frozen}


def format_gold_answer(schema: str, gold: dict) -> str:
    if schema != "pdb_structure":
        raise ValueError("This repository supports only sequence-to-structure prediction")
    return "FINAL:\n" + gold["reference_pdb"]
