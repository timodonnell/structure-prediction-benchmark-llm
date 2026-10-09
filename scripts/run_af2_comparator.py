"""Run confidence-selected AlphaFold2 comparators; import without access to gold for selection."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from pdbthink.dataset import load_dataset
from pdbthink.evaluation.runner import dataset_fingerprint
from pdbthink.evaluation.score import score_run
from pdbthink.prediction.pdb import sanitise_reference
from pdbthink.schemas import EvaluationResult
from pdbthink.util import REPO_ROOT, sha256_bytes, write_json, write_jsonl

ROOT = REPO_ROOT / "runs/alphafold2_comparator_v1"
SCORE_ROOT = REPO_ROOT / "scores/alphafold2_comparator_v1"
DATASETS = {
    "designed": REPO_ROOT / "data/datasets/prediction_pilot_v1",
    "experimental": REPO_ROOT / "data/datasets/prediction_pdb_v1",
}
CONFIG = REPO_ROOT / "configs/alphafold2_comparator_v1.yaml"
WEIGHTS = Path(os.environ.get("PDBTHINK_AF2_WEIGHTS", Path.home() / ".cache/colabfold"))


def configure_mode(single_sequence: bool):
    global ROOT, SCORE_ROOT, CONFIG
    stem = "alphafold2_single_sequence_v1" if single_sequence else "alphafold2_comparator_v1"
    ROOT = REPO_ROOT / "runs" / stem
    SCORE_ROOT = REPO_ROOT / "scores" / stem
    CONFIG = REPO_ROOT / "configs" / f"{stem}.yaml"


def audit_single_sequence(a3m: Path, sequence: str) -> dict:
    """Reject cached homologue alignments, allowing ColabFold's duplicate query rows."""
    records = []
    for line in a3m.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            records.append("")
        elif not records:
            raise ValueError(f"Malformed single-sequence alignment: {a3m}")
        else:
            records[-1] += line.strip()
    if not records or any(record != sequence for record in records):
        raise ValueError(f"Single-sequence comparator contains a non-query alignment row: {a3m}")
    return {"a3m_sha256": sha256_bytes(a3m.read_bytes()), "query_rows": len(records), "homologue_rows": 0}


def prepare(name):
    instances, renders = load_dataset(DATASETS[name])
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / name
    output.mkdir(parents=True, exist_ok=True)
    fasta = "".join(
        ">" + i.semantic_instance_id + "\n" + i.question_parameters["sequence"] + "\n" for i in instances
    )
    manifest = {
        "dataset_fingerprint": dataset_fingerprint(renders),
        "fasta_sha256": sha256_bytes(fasta.encode()),
        "config": config,
        "config_sha256": sha256_bytes(CONFIG.read_bytes()),
        "weights": {
            p.name: sha256_bytes(p.read_bytes()) for p in sorted((WEIGHTS / "params").glob("*ptm.npz"))
        },
    }
    path = output / "comparator_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Comparator inputs changed; create a new version")
    write_json(path, manifest)
    (output / "sequences.fasta").write_text(fasta)
    return output, instances, renders, config


def collect(name):
    output, instances, renders, config = prepare(name)
    by_id = {i.semantic_instance_id: i for i in instances}
    results, provenance = [], []
    run_id = config["comparator_version"] + "-" + dataset_fingerprint(renders)[:12]
    for render in renders:
        identifier = render.semantic_instance_id
        if not (output / "raw" / f"{identifier}.done.txt").exists():
            continue
        matches = list((output / "raw").glob(f"{identifier}_unrelaxed_rank_001_*.pdb"))
        if len(matches) != 1:
            raise ValueError(f"{identifier}: require exactly one confidence-ranked structure")
        raw = matches[0]
        instance = by_id[identifier]
        alignment_audit = None
        if config["msa_mode"] == "single_sequence":
            alignment_audit = audit_single_sequence(
                output / "raw" / f"{identifier}.a3m", instance.question_parameters["sequence"]
            )
        canonical = sanitise_reference(
            raw.read_text(),
            instance.question_parameters["sequence"],
            instance.question_parameters["metrics"]["backbone_atoms"],
        )
        results.append(
            EvaluationResult(
                run_id=run_id,
                render_id=render.render_id,
                completion_index=0,
                model_provider="local_alphafold2",
                model_id=config["model_id"],
                model_revision=config["comparator_version"],
                sampling_parameters=config,
                max_output_tokens=0,
                raw_response="FINAL:\n" + canonical,
            ).model_dump()
        )
        provenance.append(
            {
                "render_id": render.render_id,
                "raw_path": str(raw),
                "raw_sha256": sha256_bytes(raw.read_bytes()),
                "canonical_sha256": sha256_bytes(canonical.encode()),
                "selection": "internal pLDDT rank 001; never reference accuracy",
                "single_sequence_alignment_audit": alignment_audit,
            }
        )
    if not results:
        return 0
    responses = output / "responses"
    write_jsonl(responses / "results.jsonl", results)
    write_json(
        responses / "run_config.json",
        {
            "run_id": run_id,
            "dataset_fingerprint": dataset_fingerprint(renders),
            "model": config,
            "n_expected": len(renders),
            "n_results": len(results),
            "output_adapter": config["output_adapter"],
            "formatting_comparison_eligible": False,
        },
    )
    write_json(output / "import_provenance.json", provenance)
    score_run(DATASETS[name], responses, SCORE_ROOT / name)
    return len(results)


def existing_process(output):
    path = output / "pid"
    if not path.exists():
        return False
    pid = int(path.read_text())
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"colabfold_batch" in command and str(output / "raw").encode() in command
    except FileNotFoundError:
        return False


def launch(output, config):
    binary = REPO_ROOT / ".venv-af2/bin/colabfold_batch"
    packages = next((REPO_ROOT / ".venv-af2/lib").glob("python*/site-packages"))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in (packages / "nvidia").glob("*/lib"))
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_PLATFORMS"] = "cuda,cpu"
    check = subprocess.run(
        [
            str(binary.with_name("python")),
            "-c",
            'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "GPU required"',
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        raise RuntimeError(check.stderr)
    args = [
        str(binary),
        str(output / "sequences.fasta"),
        str(output / "raw"),
        "--model-type",
        config["model_type"],
        "--msa-mode",
        config["msa_mode"],
        "--num-models",
        str(config["num_models"]),
        "--model-order",
        ",".join(map(str, config["model_order"])),
        "--num-recycle",
        str(config["num_recycle"]),
        "--recycle-early-stop-tolerance",
        str(config["recycle_early_stop_tolerance"]),
        "--random-seed",
        str(config["random_seed"]),
        "--num-seeds",
        str(config["num_seeds"]),
        "--rank",
        config["rank"],
        "--data",
        str(WEIGHTS),
        "--compile-mode",
        "fast",
    ]
    if config["templates"] or config["relax"]:
        raise ValueError("This comparator version forbids templates and relaxation")
    write_json(output / "command.json", {"argv": args, "library_path": env["LD_LIBRARY_PATH"]})
    with (output / "run.log").open("a") as log:
        process = subprocess.Popen(
            args,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (output / "pid").write_text(str(process.pid) + "\n")
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--single-sequence", action="store_true", help="separate query-only AF2 comparator")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    configure_mode(args.single_sequence)
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.single_sequence and not args.collect_only:
        for name in args.datasets:
            prepare(name)
        # The already-running MSA controller predates GPU locking. Wait for its
        # controller and children, including gaps between its two panels.
        msa_root = REPO_ROOT / "runs/alphafold2_comparator_v1"
        while True:
            pid_file = msa_root / "controller.pid"
            controller_active = False
            if pid_file.exists():
                command = Path("/proc") / pid_file.read_text().strip() / "cmdline"
                try:
                    controller_active = b"scripts.run_af2_comparator" in command.read_bytes()
                except FileNotFoundError:
                    pass
            if not controller_active and not any(existing_process(msa_root / name) for name in DATASETS):
                break
            print("Single-sequence panels frozen; waiting for MSA comparator to release GPU", flush=True)
            time.sleep(60)
    for name in args.datasets:
        while not (DATASETS[name] / "renders.jsonl").exists():
            if args.collect_only:
                break
            print("Waiting for frozen dataset:", name, flush=True)
            time.sleep(60)
        if not (DATASETS[name] / "renders.jsonl").exists():
            continue
        output, _, renders, config = prepare(name)
        completed = collect(name)
        if args.collect_only:
            print(name, completed, "/", len(renders))
            continue
        while existing_process(output):
            completed = collect(name)
            print(name, "collected", completed, "/", len(renders), flush=True)
            time.sleep(60)
        if collect(name) < len(renders):
            process = launch(output, config)
            while process.poll() is None:
                print(name, "collected", collect(name), "/", len(renders), flush=True)
                time.sleep(60)
            if process.returncode:
                raise RuntimeError(f"ColabFold exited {process.returncode}; inspect {output}/run.log")
        completed = collect(name)
        if completed != len(renders):
            raise RuntimeError(f"Incomplete AF2 panel {name}: {completed}/{len(renders)}; no automatic rerun")
        print("Complete:", name, flush=True)
        print("Scores:", SCORE_ROOT / name, flush=True)


if __name__ == "__main__":
    main()
