# F02 source provenance and selection

F02 was acquired with a separate RCSB Search API query, not the FoldBench lists
used for part of pdbthink's coordinate benchmark. The frozen query required protein
entities of 40–550 residues, one deposited polymer chain, X-ray diffraction,
resolution at most 2.5 Å, and initial release from 2024-01-01 to 2026-09-10.

The response reported 7,745 matching polymer entities and returned 1,665
representatives grouped at 30% sequence identity. Fourteen representatives already
in the coordinate configurations were excluded, leaving 1,651 acquired candidates.
The acquisition manifest's `excluded_existing_entries: 1127` is the size of the
complete coordinate exclusion list, not the number removed from this query.
The configured candidate limit was 2,000, so it did not truncate the final pool.

- [Exact RCSB query](../data/prediction_sources/pdb_v1/query.json)
- [Original response](../data/prediction_sources/pdb_v1/response.json)
- [Source IDs and coordinate hashes](../data/prediction_sources/pdb_v1/sources.json)
- [Frozen coordinate-entry exclusions](../data/provenance/coordinate_exclusions.json)
- [Final screening rejections](../data/frozen/prediction_pdb_v1/rejections.jsonl)
- [Final selection and counts](../data/frozen/prediction_pdb_v1/selection.json)
- [Frozen panel manifest](../configs/prediction_pdb_v1_panel.json)

## Screening: 1,651 to 24

Candidates required one coordinate model, one L-protein entity and one observed
protein chain. Every annotated biological assembly had to have oligomeric count
one. The entire canonical entity sequence, including both termini, had to be
observed with exact residue mapping and all N/CA/C/O atoms. Nonstandard sequence
letters, backbone alternate locations or partial occupancy, nonwater ligands and
cofactors, and adjacent Cα distances outside 2.7–4.2 Å caused rejection. Synthetic
polymer sources and titles containing “de novo”, “de-novo”, “designed” or
“artificial” were excluded. The builder also rechecked source hashes, dates,
experimental method and resolution.

| First-failing rejection reason | Candidates |
| --- | ---: |
| Nonwater ligand present | 961 |
| Assembly not unambiguously monomeric | 399 |
| Unobserved or mismatched entity residues | 174 |
| Design-related title | 61 |
| Nonstandard entity sequence | 17 |
| Missing or ambiguous backbone atom | 13 |
| Synthetic polymer source | 1 |
| Reference chain break | 1 |
| **Total rejected** | **1,627** |

These counts record the first failing criterion; a rejected structure may violate
additional criteria. They sum to 1,627, leaving 24 eligible references.

## Selection: 24 to 10

The deterministic seed is 2026091101. Each candidate has a three-component feature
vector: the second and third singular values of centred Cα coordinates divided
by the first, and sequence length divided by 550. The first candidate is chosen
by seeded hash order. Subsequent candidates are considered in descending distance
to their nearest selected neighbour, with seeded hash tie-breaking. Accepted
candidates must satisfy all pairwise diversity conditions:

- TM-align similarity at most 0.65, using the larger of the two length-normalised scores.
- Reject sequence identity above 30% when paired alignment coverage is at least
  80% of the shorter sequence. Identity is identical aligned residues divided by
  the shorter sequence length; the code uses global affine-gap alignment.
- No overlap in the recorded publication fields.

There were no explicit fold-class or resolution quotas. Resolution was an
eligibility threshold, and length contributed to diversity without fixed final
length bins. The first 600-candidate screen yielded eight eligible structures;
the screen was expanded and proposed F01 length matching was dropped before F02
model answers were generated. The final F02 panel is not exactly length-matched
to F01. Final example numbering is by length, not greedy selection order.

| PDB ID | Chain | Residues | Initial release | Resolution (Å) |
| --- | --- | ---: | --- | ---: |
| 9PTF | A | 51 | 2026-08-05 | 1.200 |
| 8JNT | A | 77 | 2024-06-12 | 2.201 |
| 8Z3H | A | 132 | 2025-04-23 | 1.500 |
| 8T7K | A | 156 | 2024-06-26 | 1.780 |
| 26JK | A | 158 | 2026-05-27 | 1.550 |
| 8K4I | A | 163 | 2024-04-17 | 1.480 |
| 8HWO | A | 328 | 2024-01-17 | 1.960 |
| 9LM8 | A | 381 | 2026-01-28 | 2.450 |
| 9FPC | A | 384 | 2025-06-25 | 1.965 |
| 9LVN | A | 502 | 2026-01-21 | 1.880 |

## Training overlap

The date filter establishes recency, not a verified training holdout. There was
no model-specific training-cutoff verification or exclusion against each LLM's
training corpus. Sequences, homologues and earlier structures may already have
been public. The 30% clustering controls pool redundancy, not train/test overlap.
The now-published source inputs make this a reproducible public evaluation set;
a fresh secret holdout would require a separate version and selection process.
