"""Frozen fold-v1 sequence-only prompt; shared by F01 and F02."""

from ..util import stable_hash

PREDICTION_PROMPT_VERSION = "fold-v1"


PREDICTION_SYSTEM_PROMPT = """Predict a protein structure from its amino-acid sequence.
Use only the supplied sequence and your own knowledge. Do not use tools,
external resources or an external structure predictor. You may reason before
your answer. Put the complete machine-readable structure after a line FINAL:.
The final block must contain only the requested PDB records."""


PREDICTION_QUESTION = """Predict one plausible folded monomeric backbone for this sequence.
Return a single-chain PDB with exactly the atoms N, CA, C and O for each residue.
Use chain A, consecutive residue numbers 1 through {length}, standard three-letter
residue names matching the sequence, and unique consecutive atom serial numbers.
Use fixed-column ATOM records, Cartesian coordinates in Angstrom to three decimal
places, occupancy 1.00, B-factor 0.00, and the correct element in columns 77-78.
End with an END record. Do not omit residues, insert extra residues, include
side-chain atoms, alternate locations, multiple models, or connectivity records.
The global orientation and translation do not matter; do not reflect the structure.

Sequence ({length} residues):
{sequence}

After FINAL:, provide the complete PDB, not a description, algorithm or code to
generate it. Do not abbreviate coordinate records with ellipses."""


def prediction_prompt_fingerprint() -> str:
    """Hash the separate sequence-prediction contract without mixing tracks."""
    return stable_hash(PREDICTION_PROMPT_VERSION, PREDICTION_SYSTEM_PROMPT, PREDICTION_QUESTION)[:16]
