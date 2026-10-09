"""Standalone commands for frozen F01/F02 builds and evaluations."""

import argparse
import json
from pathlib import Path

import yaml

from .util import REPO_ROOT
from .validate import PANELS, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Rebuild one frozen panel from source inputs")
    build.add_argument("--panel", choices=PANELS, required=True)
    build.add_argument("--output", type=Path)
    validate = commands.add_parser("validate", help="Check prompt, label and manifest hashes")
    validate.add_argument("--dataset", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Run a versioned model configuration")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model-config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--resume", action="store_true")
    score = commands.add_parser("score", help="Score stored responses offline")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        output = args.output or REPO_ROOT / "data/datasets" / PANELS[args.panel][0]
        if args.panel == "F01":
            from .prediction.dataset import build_prediction_dataset

            sources = REPO_ROOT / "data/prediction_sources/exp278"
            candidates = sorted(sources.glob("l*-screen/candidates.parquet"))
            if len(candidates) != 6:
                raise ValueError("Fetch the pinned input archive first: python -m scripts.fetch_inputs")
            build_prediction_dataset(REPO_ROOT / "configs/prediction_v1.yaml", candidates,
                                     sources / "retention.csv", sources / "quality.csv", output)
        else:
            from .prediction.pdb_dataset import build_pdb_prediction_dataset

            build_pdb_prediction_dataset(REPO_ROOT / "configs/prediction_pdb_v1.yaml",
                                        REPO_ROOT / "data/prediction_sources/pdb_v1/sources.json", output)
        result = validate_dataset(output)
    elif args.command == "validate":
        result = validate_dataset(args.dataset)
    elif args.command == "evaluate":
        from .evaluation.runner import EvaluationRunner, ModelConfig

        validate_dataset(args.dataset)
        raw_config = yaml.safe_load(args.model_config.read_text())
        if raw_config.get("provider") != "mock" and not raw_config.get("max_output_tokens"):
            raise ValueError("Set the largest supported output budget that fits the exact prompt")
        model = ModelConfig.load(args.model_config)
        result = EvaluationRunner(args.dataset, model, args.output, resume=args.resume).run()
    else:
        from .evaluation.score import score_run

        validate_dataset(args.dataset)
        result = score_run(args.dataset, args.responses, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
