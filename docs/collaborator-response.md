1. **Where did the 1,651 F02 candidates come from, and was there a date cutoff?**

F02 used a separate RCSB PDB query, rather than the FoldBench lists used for part of the main coordinate benchmark. We required initial release between **1 January 2024 and 10 September 2026**, X-ray diffraction at **≤2.5 Å**, protein length **40–550 residues**, and one deposited polymer chain. The frozen response contained 7,745 matching polymer entities, grouped into 1,665 representatives at 30% sequence identity. Removing 14 entries already present in the coordinate-dataset configurations left **1,651 candidates**.

The leakage concern is real: **this is not a verified training-set holdout**. We did not verify model-specific training cutoffs or exclude homologues against each LLM's training corpus. Sequences, homologues and related structures may have been public before the selected PDB release. The 30% clustering reduces redundancy within the pool; it does not establish separation from training data. We should describe F02 as a recent experimental-structure pilot with unresolved contamination risk. The [exact query and selection audit](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/docs/f02-provenance.md) are now available.

2. **What was the strict screening, and how were the final ten selected?**

Candidates required one coordinate model, one L-protein entity and one observed protein chain, with every annotated biological assembly monomeric. The **entire canonical entity sequence**, including both termini, had to be observed with exact residue mapping and complete N/CA/C/O backbone atoms. We rejected nonstandard sequence letters, ambiguous backbone alternate locations or partial occupancy, nonwater ligands/cofactors, and adjacent Cα distances outside 2.7–4.2 Å. Synthetic-source records and design-related titles were excluded as well.

The [rejection log](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/data/frozen/prediction_pdb_v1/rejections.jsonl) accounts for all exclusions:

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

These are first-failing reasons, so a structure may violate more than one criterion. **24 candidates passed.**

The final ten were selected before F02 predictions using a deterministic seeded procedure favouring **overall Cα shape and length diversity**. Its features were two singular-value ratios describing shape and length divided by 550. Selected pairs also had to have TM-align similarity ≤0.65, avoid sequence identity >30% at ≥80% coverage of the shorter sequence, and have no shared recorded publication fields.

There were **no explicit fold-class or resolution quotas**. Resolution was a quality threshold; length contributed to diversity without fixed final bins. The initial screen of 600 candidates yielded only eight eligible structures, so the pool was expanded and proposed F01 length matching was dropped before generating F02 predictions. **F02 is not exactly length-matched to F01.** The [selection code](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/src/pdbthink/prediction/pdb_dataset.py) and [configuration](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/configs/prediction_pdb_v1.yaml) specify the procedure.

3. **PDB IDs**

All ten use entity 1, chain A. These are the entries in the [frozen panel manifest](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/configs/prediction_pdb_v1_panel.json):

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

4. **Can we access the code?**

Yes—we have now published the standalone benchmark at **[timodonnell/structure-prediction-benchmark-llm](https://github.com/timodonnell/structure-prediction-benchmark-llm)**. It includes:

- [Shared F01/F02 prompt](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/src/pdbthink/prompts/library.py), [F01 builder](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/src/pdbthink/prediction/dataset.py) and [F02 builder](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/src/pdbthink/prediction/pdb_dataset.py).
- [PDB syntax, sequence and completeness validator](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/src/pdbthink/prediction/pdb.py).
- [AF2 runner and output adapter](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/scripts/run_af2_comparator.py), with separate [MSA-assisted](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/configs/alphafold2_comparator_v1.yaml) and [single-sequence](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/configs/alphafold2_single_sequence_v1.yaml) configurations.
- [Structural scoring](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/src/pdbthink/prediction/metrics.py), [aggregation](https://github.com/timodonnell/structure-prediction-benchmark-llm/blob/v0.1.0/src/pdbthink/prediction/report.py), tests, frozen manifests and source-selection records.
- A [versioned source-input release](https://github.com/timodonnell/structure-prediction-benchmark-llm/releases/tag/v0.1.0), with hashes and commands to rebuild both panels independently of the original pdbthink checkout.

Both panels were rebuilt and verified against the original prompt/reference fingerprints and manifest hashes. Mock-gold checks give primary Cα lDDT and TM-score exactly 1.0 in both panels. No new LLM or AF2 inference was needed for this extraction.

Both AF2 routes disable templates and select by internal pLDDT. Their output is normalised into backbone PDB format, so their formatting success is not directly comparable to raw LLM PDB writing. F01 measures agreement with computational design references; F02 uses experimental references. Their scores remain separate.
