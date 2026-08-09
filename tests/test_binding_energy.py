import pytest

from fold.binding_energy import detect_ligand_chain, split_receptor_and_ligand_centroid

# Fixed-column PDB lines (columns verified against a real gemmi-converted
# OpenFold3 output in structure_convert.py's manual test).
PROTEIN_LINE = "ATOM      1  N   GLY A   1     -30.594  19.035  11.772  1.00 48.03           N  \n"
LIGAND_LINE_1 = "HETATM 3485  C1  LIG Z           9.000  10.000   1.000  1.00 83.22           C  \n"
LIGAND_LINE_2 = "HETATM 3486  C2  LIG Z          11.000  12.000   3.000  1.00 90.10           C  \n"


def test_detect_ligand_chain(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PROTEIN_LINE + LIGAND_LINE_1 + LIGAND_LINE_2)

    assert detect_ligand_chain(pdb) == "Z"


def test_detect_ligand_chain_no_hetatm_raises(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PROTEIN_LINE)

    with pytest.raises(ValueError, match="no HETATM"):
        detect_ligand_chain(pdb)


def test_detect_ligand_chain_multiple_chains_raises(tmp_path):
    other_chain_line = LIGAND_LINE_1.replace(" Z ", " Y ", 1)
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PROTEIN_LINE + LIGAND_LINE_1 + other_chain_line)

    with pytest.raises(ValueError, match="exactly one ligand chain"):
        detect_ligand_chain(pdb)


def test_split_receptor_and_ligand_centroid(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(PROTEIN_LINE + LIGAND_LINE_1 + LIGAND_LINE_2)
    receptor_path = tmp_path / "receptor.pdb"

    out_path, centroid = split_receptor_and_ligand_centroid(pdb, "Z", receptor_path)

    assert out_path == receptor_path
    receptor_text = receptor_path.read_text()
    assert "GLY" in receptor_text
    assert "LIG" not in receptor_text
    assert centroid == pytest.approx((10.0, 11.0, 2.0))
