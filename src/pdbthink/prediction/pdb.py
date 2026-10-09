"""Strict validation of the explicitly requested backbone-PDB dialect.

PDB syntax, completeness/sequence fidelity and geometric plausibility are
different properties. A syntactically valid but distorted structure remains
eligible for accuracy metrics; geometry diagnostics do not censor bad models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import gemmi
import numpy as np


@dataclass
class Backbone:
    syntax_valid: bool = False
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    coordinates: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    residues: dict[int, str] = field(default_factory=dict)
    atom_records: int = 0
    valid_atom_records: int = 0

    @property
    def ca(self) -> np.ndarray:
        return np.array([self.coordinates[(i, "CA")] for i in sorted(self.residues)])


def extract_pdb(response: str) -> str:
    """Preserve fixed columns while accepting one final block or fenced PDB."""
    markers = list(re.finditer(r"(?im)^\s*(?:\*\*)?FINAL(?:\*\*)?\s*:?[ \t]*$", response))
    text = response[markers[-1].end() :] if markers else response
    lines = text.strip("\r\n ").splitlines()
    if lines and re.fullmatch(r"```(?:pdb)?", lines[0].strip(), flags=re.I):
        if len(lines) < 2 or lines[-1].strip() != "```":
            return text
        lines = lines[1:-1]
    return "\n".join(lines) + "\n"


def validate_pdb(text: str, sequence: str, atoms: list[str]) -> Backbone:
    result = Backbone()
    syntax_errors, protocol_errors = [], []
    serials, keys, chains = [], set(), set()
    residue_order = []
    end_seen = False
    for line in text.splitlines():
        if not line.strip():
            continue
        if end_seen:
            syntax_errors.append("record_after_end")
            continue
        record = line[:6].strip()
        if record == "END":
            end_seen = True
            if line[6:].strip():
                syntax_errors.append("invalid_end_record")
            continue
        if record != "ATOM":
            syntax_errors.append("unsupported_record")
            continue
        result.atom_records += 1
        try:
            if len(line) < 78 or line[:6] != "ATOM  ":
                raise ValueError
            serial, number = int(line[6:11]), int(line[22:26])
            name, residue, chain = line[12:16].strip(), line[17:20], line[21]
            xyz = np.array([float(line[i : i + 8]) for i in (30, 38, 46)])
            occ, bf = float(line[54:60]), float(line[60:66])
            if not np.isfinite([*xyz, occ, bf]).all() or serial < 1 or not name:
                raise ValueError
            if line[76:78].strip() != name[0]:
                raise ValueError
            aa = gemmi.find_tabulated_residue(residue).one_letter_code
            if not aa or aa not in "ACDEFGHIKLMNPQRSTVWY":
                raise ValueError
        except (ValueError, IndexError):
            syntax_errors.append("invalid_atom_record")
            continue
        result.valid_atom_records += 1
        chains.add(chain)
        serials.append(serial)
        residue_order.append(number)
        key = (number, name)
        if key in keys:
            protocol_errors.append("duplicate_atom")
        keys.add(key)
        if number in result.residues and result.residues[number] != aa:
            protocol_errors.append("inconsistent_residue_name")
        result.residues[number] = aa
        result.coordinates[key] = xyz
        if line[16] != " " or line[26] != " ":
            protocol_errors.append("alternate_location_or_insertion_code")
        if occ != 1.0 or bf != 0.0:
            protocol_errors.append("occupancy_or_bfactor_not_requested")
        if any(not re.fullmatch(r" *-?\d+\.\d{3}", line[i : i + 8]) for i in (30, 38, 46)):
            protocol_errors.append("coordinate_precision")
    if not end_seen:
        syntax_errors.append("missing_end")
    if result.atom_records == 0:
        syntax_errors.append("no_atoms")
    result.syntax_valid = not syntax_errors
    if chains != {"A"}:
        protocol_errors.append("chain_must_be_A")
    if serials != list(range(1, len(serials) + 1)):
        protocol_errors.append("atom_serial_numbering")
    if residue_order != sorted(residue_order):
        protocol_errors.append("residue_record_order")
    if sorted(result.residues) != list(range(1, len(sequence) + 1)):
        protocol_errors.append("residue_numbering_or_coverage")
    if "".join(result.residues[i] for i in sorted(result.residues)) != sequence:
        protocol_errors.append("sequence_mismatch")
    expected = {(i, name) for i in range(1, len(sequence) + 1) for name in atoms}
    if expected - keys:
        protocol_errors.append("missing_backbone_atoms")
    if keys - expected:
        protocol_errors.append("extra_atoms")
    result.errors = sorted(set(syntax_errors + protocol_errors))
    result.valid = not result.errors
    return result


def format_backbone(sequence: str, coordinates: np.ndarray, atoms: list[str]) -> str:
    """Emit canonical records from coordinates in residue-major atom order."""
    array = np.asarray(coordinates, dtype=float).reshape(len(sequence), len(atoms), 3)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite_coordinates")
    lines, serial = [], 0
    for index, aa in enumerate(sequence, 1):
        residue = gemmi.expand_one_letter(aa, gemmi.ResidueKind.AA)
        for name, xyz in zip(atoms, array[index - 1], strict=True):
            serial += 1
            fields = [f"{value:8.3f}" for value in xyz]
            if any(len(s) != 8 for s in fields):
                raise ValueError("coordinate_overflow")
            lines.append(
                f"ATOM  {serial:5d} {name:^4s} {residue:3s} A{index:4d}    "
                + "".join(fields)
                + f"{1.0:6.2f}{0.0:6.2f}          {name[0]:>2s}"
            )
    return "\n".join([*lines, "END"]) + "\n"


def sanitise_reference(pdb: str, sequence: str, atoms: list[str]) -> str:
    structure = gemmi.read_pdb_string(pdb)
    if len(structure) != 1 or len(structure[0]) != 1:
        raise ValueError("reference_not_single_chain")
    residues = list(structure[0][0])
    actual = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)
    if actual != sequence:
        raise ValueError("reference_sequence_mismatch")
    coords = []
    for residue in residues:
        for name in atoms:
            group = residue[name]
            if len(group) != 1:
                raise ValueError("reference_backbone_incomplete_or_ambiguous")
            coords.append(list(group[0].pos))
    return format_backbone(sequence, np.asarray(coords), atoms)
