import pytest

from fold.inputs import CofoldInput, parse_input_file, parse_sequence_smiles_file


def test_parses_basic_two_line_file(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("SEQUENCE: MLSRLFRMHGLFVASHPWEVIVG\nSMILES: CC(C)C1=NC(=NC(=C1)C2=CC=C(C=C2)F)N\n")

    sequence, smiles = parse_sequence_smiles_file(f)

    assert sequence == "MLSRLFRMHGLFVASHPWEVIVG"
    assert smiles == "CC(C)C1=NC(=NC(=C1)C2=CC=C(C=C2)F)N"


def test_order_of_lines_does_not_matter(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("SMILES: CCO\nSEQUENCE: MLS\n")

    sequence, smiles = parse_sequence_smiles_file(f)

    assert sequence == "MLS"
    assert smiles == "CCO"


def test_lowercase_prefix_and_extra_whitespace(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("  sequence:   MLS  \n  smiles:  CCO  \n")

    sequence, smiles = parse_sequence_smiles_file(f)

    assert sequence == "MLS"
    assert smiles == "CCO"


def test_blank_lines_are_ignored(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("\nSEQUENCE: MLS\n\nSMILES: CCO\n\n")

    sequence, smiles = parse_sequence_smiles_file(f)

    assert sequence == "MLS"
    assert smiles == "CCO"


def test_missing_smiles_raises(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("SEQUENCE: MLS\n")

    with pytest.raises(ValueError, match="SEQUENCE.*SMILES"):
        parse_sequence_smiles_file(f)


def test_missing_sequence_raises(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("SMILES: CCO\n")

    with pytest.raises(ValueError):
        parse_sequence_smiles_file(f)


def test_empty_file_raises(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("")

    with pytest.raises(ValueError):
        parse_sequence_smiles_file(f)


def test_parse_input_file_dimer_without_smiles(tmp_path):
    f = tmp_path / "dimer.txt"
    f.write_text("SEQUENCE: MLS\nSEQUENCE_B: TET\n")

    parsed = parse_input_file(f)

    assert parsed == CofoldInput(sequence="MLS", smiles="", sequence_b="TET")
    assert parsed.is_dimer


def test_parse_input_file_with_all_three_lines(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("SEQUENCE: MLS\nSMILES: CCO\nSEQUENCE_B: TET\n")

    parsed = parse_input_file(f)

    assert parsed == CofoldInput(sequence="MLS", smiles="CCO", sequence_b="TET")
    assert not parsed.is_dimer


def test_parse_input_file_requires_smiles_or_second_sequence(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("SEQUENCE: MLS\n")

    with pytest.raises(ValueError, match="SMILES.*SEQUENCE_B"):
        parse_input_file(f)


def test_parse_sequence_smiles_file_still_rejects_dimer_file(tmp_path):
    """The RF3/ESMFold path doesn't take a second sequence: a file with only
    SEQUENCE + SEQUENCE_B is fine for parse_input_file but not for the
    protein+ligand-only parser."""
    f = tmp_path / "dimer.txt"
    f.write_text("SEQUENCE: MLS\nSEQUENCE_B: TET\n")

    with pytest.raises(ValueError, match="SEQUENCE.*SMILES"):
        parse_sequence_smiles_file(f)
