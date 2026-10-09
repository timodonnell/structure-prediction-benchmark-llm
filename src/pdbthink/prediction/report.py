"""Sequence-prediction summaries keep validity and conditional accuracy visible."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _interval(values, seed=20260910, samples=4000):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": None, "ci_low": None, "ci_high": None, "n_examples": 0}
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_examples": len(array),
    }


def summarise_prediction(rows, samples=4000):
    """Average repeats within examples before resampling independent examples."""
    references = {"experimental_pdb" if r.get("question_family") == "F02" else "esmfold" for r in rows}
    if len(references) > 1:
        raise ValueError("experimental and designed prediction panels must be reported separately")
    primary_reference = next(iter(references), "esmfold")
    groups = defaultdict(list)
    for row in rows:
        groups[row["semantic_instance_id"]].append(row)

    def per_example(function):
        return [np.mean([function(row) for row in group]) for group in groups.values()]

    def valid(row):
        return bool(row["detail"].get("accuracy_eligible")) and not any(
            row.get(k, False) for k in ("truncated", "refusal", "api_error")
        )

    result = {
        "primary_reference": primary_reference,
        "formatting_comparison_eligible": not any(
            r.get("model_provider") == "local_alphafold2" for r in rows
        ),
        "n_results": len(rows),
        "n_examples": len(groups),
        "n_accuracy_eligible": sum(valid(r) for r in rows),
        "model_id": rows[0]["model_id"] if rows else None,
        "syntax_valid": _interval(
            per_example(lambda r: float(r["detail"].get("syntax_valid", False))), samples=samples
        ),
        "protocol_valid": _interval(
            per_example(lambda r: float(r["detail"].get("protocol_valid", valid(r)))), samples=samples
        ),
        "accuracy_eligible_rate": _interval(per_example(lambda r: float(valid(r))), samples=samples),
        "accuracy": {},
    }
    for reference in ("esmfold", "proteina") if primary_reference == "esmfold" else (primary_reference,):
        result["accuracy"][reference] = {}
        for metric in ("lddt_ca", "tm_score", "gdt_ts", "gdt_ha", "rmsd_ca"):
            conditional = []
            unconditional = []
            for group in groups.values():
                values = [
                    r["detail"][reference][metric] for r in group if valid(r) and reference in r["detail"]
                ]
                if values:
                    conditional.append(float(np.mean(values)))
                if metric != "rmsd_ca":
                    unconditional.append(sum(values) / len(group))
            result["accuracy"][reference][metric] = {
                "conditional": _interval(conditional, samples=samples),
                "zero_for_invalid": _interval(unconditional, samples=samples)
                if metric != "rmsd_ca"
                else None,
            }
    return result


def prediction_markdown(runs):
    lines = [
        "",
        "## Sequence-to-structure pilot (separate track)",
        "",
        "F01 measures agreement with computational references; F02 uses experimental PDB coordinates. "
        "Intervals resample examples after averaging repeats; ten examples support an exploratory pilot. "
        "Conditional scores exclude invalid outputs and can favour a model that emits few valid PDBs. "
        "Use the score with invalid outputs counted as zero for model comparisons. "
        "AlphaFold2 outputs marked † undergo deterministic backbone format normalisation; "
        "their syntax rates are not a raw text-formatting comparison with LLMs.",
        "",
        "| Model | Reference | Examples | Eligible / completions | PDB syntax | Protocol valid | "
        "Reference lDDT, all outputs (95% CI) | Conditional lDDT |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]

    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"

    for run in runs:
        model_label = run["model_id"] + ("" if run.get("formatting_comparison_eligible", True) else " †")
        reference = run.get("primary_reference", "esmfold")
        all_scores = run["accuracy"][reference]["lddt_ca"]["zero_for_invalid"]
        conditional = run["accuracy"][reference]["lddt_ca"]["conditional"]
        lines.append(
            f"| {model_label} | {reference} | {run['n_examples']} | "
            f"{run['n_accuracy_eligible']} / {run['n_results']} | "
            f"{fmt(run['syntax_valid']['mean'])} | {fmt(run['protocol_valid']['mean'])} | "
            f"{fmt(all_scores['mean'])} [{fmt(all_scores['ci_low'])}, {fmt(all_scores['ci_high'])}] | "
            f"{fmt(conditional['mean'])} |"
        )
        lines += [
            "",
            f"### {run['model_id']}: structural agreement",
            "",
            "| Reference | Metric | Conditional mean | Examples with valid output | Invalid = zero |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for reference, metrics in run["accuracy"].items():
            for metric, stats in metrics.items():
                c = stats["conditional"]
                u = stats["zero_for_invalid"]
                lines.append(
                    f"| {reference} | {metric} | {fmt(c['mean'])} | {c['n_examples']} | "
                    f"{fmt(u['mean']) if u else 'not defined'} |"
                )
    return "\n".join(lines) + "\n"
