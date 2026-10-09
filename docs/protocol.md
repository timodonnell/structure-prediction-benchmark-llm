# Scientific protocol and extraction notes

The original experimental extension was specified on 11 September 2026 UTC.
The frozen sequence-only prompt is `fold-v1`. F01 uses
`prediction-v1.0.0`; F02 uses `prediction-pdb-v1.0.0`. Versioned metric settings
are in `configs/prediction_v1.yaml` and shared between panels. Extraction into
this repository does not change these definitions or reinterpret historical runs.

F01 contains ten computationally designed sequences from the completed MarinFold
exp278 pilot screen, pinned to revision
`cba806f10f05a8d34dbf307f705bccf028f2c3f0`. Selection requires the upstream
reference screen, ESMFold pLDDT at least 70, self-consistency RMSD at most 2 Å,
complete references and pairwise diversity. Length quotas are two each at 60,
100, 200 and 300 residues, and one each at 400 and 500. Its primary reference is
the versioned ESMFold refold; original Proteina Cα coordinates are a secondary
reference. Neither reference is experimental ground truth.

F02 uses complete experimental monomers. See [the source-selection audit](f02-provenance.md).
It is not exactly length-matched to F01. Length, topology, source selection and
sequence familiarity all confound a designed-versus-experimental comparison.

The LLM sees only sequence and output instructions. PDB IDs, source descriptions,
dates, publications, B-factors and reference coordinates are not in its prompt.
This prevents prompt-side source disclosure; it does not remove information
already present in pretrained weights.

## Contamination and interpretation

F02 initial release dates were restricted to 2024-01-01 through 2026-09-10.
There was no model-specific training-cutoff verification or homology exclusion
against each model's training corpus. Candidate clustering at 30% sequence
identity reduces pool redundancy, not overlap with training data. Sequences,
homologues and earlier structures may predate the reference PDB release.
The source-input release is publicly accessible for reproducibility and cannot
establish a future secret holdout. The canary is an exclusion marker, not proof
of training-set absence. Ten selected structures per panel are an exploratory
pilot; no population-wide model ranking follows from this sample.

The MSA-assisted AF2 route has evolutionary sequence information unavailable to
the sequence-only LLMs. The single-sequence AF2 route is a separate exploratory
ablation added after observing the MSA-assisted F01 results. Templates are disabled
in both routes. This prevents direct template retrieval during inference but
does not remove pretrained structural knowledge. The ablation does not equalise
training data, compute or architecture. AF2's adapter validates complete sequence
coverage and formats its backbone; its syntax rate is not an LLM formatting result.

## What was extracted

The Python namespace and record schemas remain compatible with historical
pdbthink files. Coordinate generators, mechanistic tasks and coordinate datasets
are omitted. Shared transport/scoring helpers retain some legacy schemas, but
the standalone CLI validates and accepts only one F01 or F02 panel at a time.

The source working tree contained unpublished prediction files. An upstream Git
commit alone would not identify them, so [extraction provenance](../data/provenance/extraction.json)
records their individual content hashes. Scientific selection, prompt text,
PDB validity checks, structural metrics, aggregation and AF2 inference settings
are preserved. New integration consists of the standalone CLI and validation,
input download, packaging and CI. Paths in archived cache metadata are relative;
source coordinates and F01 candidate inputs retain their original bytes.
The AF2 runner writes scores without invoking the original report-site generator.
The PDB acquisition script uses the frozen coordinate-entry exclusion list instead
of depending on the original coordinate benchmark configurations.

The original manifest withholding notice is retained verbatim for hash identity.
Its historical `structural-reasoning build` command is superseded here by
`structure-prediction-benchmark build --panel F01` or `--panel F02`.
Generated prompts and gold files remain untracked. The release archive contains
the raw source inputs needed to recompute them; this is reproducibility, not a
claim that the references are inaccessible.

The release does not include historical LLM responses, credentials, proprietary
model weights, or newly launched model evaluations. Original AF2 environment
versions are recorded, but AF2 inference is not rerun by the extraction tests.

## Verification

`build` verifies each panel against its original frozen membership, exact
prompt-and-reference fingerprint and instance/render manifest hashes. Unit tests
cover missing atoms, malformed PDBs, sequence mismatch, rigid transforms,
reflections, incomplete references, monomer screening, invalid-output accounting,
separate panel reporting, query-only AF2 alignments and explicit no-tool controls.
The mock-gold check must give primary lDDT and TM-score exactly 1.0 for both panels.
See [the extraction verification record](../data/provenance/verification.json).
