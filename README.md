# Structure prediction benchmark for tool-free LLMs

Given only an amino-acid sequence, produce a complete single-chain backbone PDB.
This repository contains the standalone F01/F02 prediction experiment extracted
from [pdbthink](https://github.com/Open-Athena/pdbthink).

| Panel | Examples | Lengths | Withheld reference |
| --- | ---: | --- | --- |
| F01 | 10 | 60–500 residues | Versioned computational designs: ESMFold refolds, with original Proteina designs as a secondary reference |
| F02 | 10 | 51–502 residues | Complete experimental X-ray PDB monomers |

**F01 is not experimental ground truth. F01 and F02 are never pooled.** These
prediction scores do not enter pdbthink's coordinate-reasoning headline.
Ten selected proteins per panel support an exploratory pilot, not a broad ranking.

The prompt requests N/CA/C/O for every residue, fixed-column PDB records, chain A,
consecutive numbering, and a `FINAL:` block. The prompt, references, selection
definitions, and metric definitions retain their original versions and hashes.

## Reproduce the frozen panels

Use Python 3.11 and a C++ compiler (`g++`). Run commands from this checkout.
Install in a separate environment: the extracted package retains the `pdbthink`
Python namespace for compatibility with historical records.

```bash
git clone https://github.com/timodonnell/structure-prediction-benchmark-llm.git
cd structure-prediction-benchmark-llm
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m scripts.install_tmscore
.venv/bin/python -m scripts.fetch_inputs
.venv/bin/python -m pdbthink.cli build --panel F01
.venv/bin/python -m pdbthink.cli build --panel F02
.venv/bin/python -m pytest -q
```

`fetch_inputs` downloads the approximately 313 MB
[v0.1.0 source-input archive](https://github.com/timodonnell/structure-prediction-benchmark-llm/releases/tag/v0.1.0)
and verifies the archive and every file against
[recorded SHA-256 hashes](data/provenance/input_release.json). It contains the
original six F01 candidate Parquet files and screen tables, plus all 1,651 F02
mmCIF files and the metadata actually used for screening. It contains no model
responses or generated gold files. The archived metadata paths are portable;
coordinate and F01 source-file bytes are unchanged.

After downloading the inputs, builds and scoring run offline. Each build checks
panel membership, exact prompt/reference fingerprint, and both manifest hashes
against the original frozen panel. The resulting datasets are written under
`data/datasets/prediction_pilot_v1` and `data/datasets/prediction_pdb_v1`.
Gold coordinates and generated prompts remain outside Git. Source inputs are
deliberately available for reproducibility; this is not a secret held-out test set.

## Check evaluation and scoring without calling a model

```bash
.venv/bin/python -m pdbthink.cli evaluate \
  --dataset data/datasets/prediction_pdb_v1 \
  --model-config configs/models/mock_gold.yaml --output runs/mock-F02
.venv/bin/python -m pdbthink.cli score \
  --dataset data/datasets/prediction_pdb_v1 \
  --responses runs/mock-F02 --output scores/mock-F02
```

Repeat with `prediction_pilot_v1` and distinct output paths for F01. The mock
provider uses the local gold label; primary reference Cα lDDT and TM-score must
both be exactly 1.0. This checks the pipeline and is not a model result.

## Run models

The evaluator supports OpenAI-compatible chat endpoints, native Anthropic and
Ollama, retains raw provider responses in the response cache, and executes no
tool calls. Start from the [model configuration template](configs/models/example.yaml)
and supply the endpoint/model and verified output limit. The template deliberately
requires those fields to be filled before it can run.

Every new request must explicitly disable tools. OpenAI-compatible requests use
`tools: []` (or explicit `null` for endpoints that reject the empty list) and
`tool_choice: none`; native Anthropic uses `tool_choice: {type: none}`, and Ollama
uses an empty tool list. OpenRouter additionally requires its web plugin disabled.
The transport rejects missing controls. No historical run is reclassified as
explicitly tool-disabled by publishing this code.

Use the largest endpoint-supported output budget that fits each exact prompt,
including reasoning tokens. Verify model-specific prompt counts and context/output
limits; if budgets differ across prompts, use the runner's `only_renders` option
with separately versioned configurations. Keep earlier capped runs, rerun when
more capacity is available, and report the comparison. Do not shorten frozen
prompts or silently reuse a capped run's identity. The transport records requested
budgets and provider token usage; it cannot discover undocumented provider clamps.

```bash
.venv/bin/python -m pdbthink.cli evaluate \
  --dataset data/datasets/prediction_pdb_v1 \
  --model-config configs/models/my-model-v1.yaml --output runs/my-model-F02-v1
.venv/bin/python -m pdbthink.cli score \
  --dataset data/datasets/prediction_pdb_v1 \
  --responses runs/my-model-F02-v1 --output scores/my-model-F02-v1
```

## AlphaFold2 comparators

There are separate **MSA-assisted** and **single-sequence** specialist routes.
Both use ColabFold 1.6.2 / alphafold-colabfold 2.3.18, five monomer pTM weight sets,
one seed, three recycles, no templates, no Amber relaxation, and selection by
internal pLDDT. Reference accuracy never selects a candidate. The single-sequence
route audits saved alignments to reject non-query rows.

For a matching Linux/CUDA environment, create `.venv-af2` with Python 3.11 and
install the [recorded original AF2 environment](requirements/alphafold2-original.txt):

```bash
uv venv --python 3.11 .venv-af2
uv pip install --python .venv-af2/bin/python -r requirements/alphafold2-original.txt
.venv-af2/bin/python -c 'from pathlib import Path; from colabfold.download import download_alphafold_params; download_alphafold_params("alphafold2_ptm", Path.home() / ".cache/colabfold")'
.venv/bin/python -m scripts.run_af2_comparator
.venv/bin/python -m scripts.run_af2_comparator --single-sequence
```

A compatible NVIDIA GPU/driver is required by this runner. Set
`PDBTHINK_AF2_WEIGHTS` for a different weights directory. The MSA route contacts the
ColabFold MMseqs2 service and retains its actual alignments; the live MSA database
may change. Do not expect a newly retrieved MSA to reproduce historical AF2 inputs.
AF2 predictions were not rerun as part of this code extraction.

Both routes normalise AF2 output into the requested backbone format. Consequently
AF2 syntax success is not a raw text-formatting comparison with LLM outputs.

## Methods and code map

| Component | Source |
| --- | --- |
| Shared sequence-only prompt | [Prompt library](src/pdbthink/prompts/library.py) |
| F01 source screening and selection | [F01 builder](src/pdbthink/prediction/dataset.py) |
| F02 RCSB query and acquisition | [Acquisition script](scripts/prepare_pdb_prediction.py) |
| F02 completeness screening and selection | [F02 builder](src/pdbthink/prediction/pdb_dataset.py) |
| PDB syntax and protocol validation | [Validator](src/pdbthink/prediction/pdb.py) |
| Structural metrics | [Scoring implementation](src/pdbthink/prediction/metrics.py) |
| Invalid-output accounting and bootstrap summaries | [Reporting](src/pdbthink/prediction/report.py) |
| AF2 execution, confidence selection and adapter | [Runner](scripts/run_af2_comparator.py) |
| Query snapshots, exclusions and rejection counts | [F02 provenance](docs/f02-provenance.md) |
| Scientific limitations and extraction details | [Protocol](docs/protocol.md) |

The primary metric is mean per-residue Cα lDDT. Secondary metrics are fixed-index
TM-score, GDT-TS, GDT-HA and Cα RMSD. Formatting/completeness eligibility and
geometric plausibility are distinct: a complete but distorted structure still
receives accuracy metrics. Report conditional accuracy and bounded scores with
invalid outputs counted as zero; RMSD does not use zero for invalid outputs.
Average repeated attempts within each sequence before bootstrap resampling.

The code is Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE), and the retained
notices in [TMscore.cpp](third_party/TMscore.cpp). Benchmark source inputs are
evaluation material and should be excluded from training corpora; see [CANARY.md](CANARY.md).
