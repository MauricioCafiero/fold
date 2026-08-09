"""Parse a plain-text protein+ligand input file for the cofolding CLIs.

Expected format, one field per line, order doesn't matter:

    SEQUENCE: MLSRLFRMHGLFVASHPWEVIVG...
    SMILES: CC(C)C1=NC(=NC(=C1/C=C/[C@H]...
"""

from pathlib import Path

_SEQUENCE_PREFIX = "SEQUENCE:"
_SMILES_PREFIX = "SMILES:"


def parse_sequence_smiles_file(path: str | Path) -> tuple[str, str]:
    """Read a SEQUENCE:/SMILES: text file, return (sequence, smiles)."""
    path = Path(path)
    sequence: str | None = None
    smiles: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(_SEQUENCE_PREFIX):
            sequence = line[len(_SEQUENCE_PREFIX) :].strip()
        elif upper.startswith(_SMILES_PREFIX):
            smiles = line[len(_SMILES_PREFIX) :].strip()

    if not sequence or not smiles:
        raise ValueError(
            f"{path} must contain a non-empty 'SEQUENCE:' line and 'SMILES:' line"
        )
    return sequence, smiles
