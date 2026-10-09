"""Experimental references preserve full sequence coverage and explicit ambiguity rejection."""

from pathlib import Path

import gemmi
import pytest
import yaml

from pdbthink.prediction.pdb_dataset import experimental_backbone
from pdbthink.util import REPO_ROOT


@pytest.fixture
def experimental_cif(tmp_path):
    structure = gemmi.read_structure(str(Path(__file__).parent / "fixtures/crambin_processed.pdb"))
    structure.setup_entities()
    residues = list(structure[0][0])
    for n, r in enumerate(residues, 1):
        r.label_seq = n
        r.entity_type = gemmi.EntityType.Polymer
    structure.entities[0].full_sequence = [r.name for r in residues]
    document = structure.make_mmcif_document()
    block = document.sole_block()
    sequence = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)
    block.find_loop("_entity_poly.entity_id").get_loop().add_columns(
        ["_entity_poly.pdbx_seq_one_letter_code_can"],
        sequence,
    )
    block.set_pair("_pdbx_struct_assembly.id", "1")
    block.set_pair("_pdbx_struct_assembly.oligomeric_count", "1")
    path = tmp_path / "reference.cif"
    document.write_file(str(path))
    config = yaml.safe_load((REPO_ROOT / "configs/prediction_v1.yaml").read_text())
    return path, structure.entities[0].name, config


def test_complete_experimental_backbone_is_valid(experimental_cif):
    path, entity, config = experimental_cif
    row = experimental_backbone(path, entity, config["metrics"], config["geometry_diagnostics"])
    assert len(row["sequence"]) == len(row["ca"]) == 46
    assert len(row["author_residue_ids"]) == 46


def test_unobserved_residues_are_not_dropped_from_sequence(experimental_cif):
    path, entity, config = experimental_cif
    document = gemmi.cif.read_file(str(path))
    block = document.sole_block()
    table = block.find("_atom_site.", ["label_seq_id"])
    for n in reversed(range(len(table))):
        if table[n][0] == "12":
            table.remove_row(n)
    document.write_file(str(path))
    with pytest.raises(ValueError, match="unobserved_or_mismatched"):
        experimental_backbone(path, entity, config["metrics"], config["geometry_diagnostics"])


def test_multimeric_assembly_is_rejected(experimental_cif):
    path, entity, config = experimental_cif
    document = gemmi.cif.read_file(str(path))
    document.sole_block().set_pair("_pdbx_struct_assembly.oligomeric_count", "2")
    document.write_file(str(path))
    with pytest.raises(ValueError, match="assembly_not_unambiguously_monomeric"):
        experimental_backbone(path, entity, config["metrics"], config["geometry_diagnostics"])
