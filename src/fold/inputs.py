"""Parse a plain-text cofolding input file for the CLIs.

Expected format, one field per line, order doesn't matter:

    SEQUENCE: MLSRLFRMHGLFVASHPWEVIVG...
    SMILES: CC(C)C1=NC(=NC(=C1/C=C/[C@H]...
    SEQUENCE_B: TETSSHKAHTEAQVINTFDGV...   (optional, dimer partner)

SEQUENCE plus SMILES is a protein+ligand query; SEQUENCE plus SEQUENCE_B
is a protein+protein (dimer) query; all three cofold two proteins with a
ligand. A SMILES line may be omitted when SEQUENCE_B is present.
"""

from dataclasses import dataclass
from pathlib import Path

_SEQUENCE_PREFIX = "SEQUENCE:"
_SEQUENCE_B_PREFIX = "SEQUENCE_B:"
_SMILES_PREFIX = "SMILES:"


@dataclass
class CofoldInput:
    """Chains parsed from an input file (subset of the given lines)."""

    sequence: str
    smiles: str = ""
    sequence_b: str = ""

    @property
    def is_dimer(self) -> bool:
        """True for a bare protein+protein query (no ligand)."""
        return bool(self.sequence_b) and not self.smiles


def parse_input_file(path: str | Path) -> CofoldInput:
    """Read a SEQUENCE:/SEQUENCE_B:/SMILES: text file into a CofoldInput."""
    path = Path(path)
    sequence: str | None = None
    sequence_b: str | None = None
    smiles: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(_SEQUENCE_B_PREFIX):
            sequence_b = line[len(_SEQUENCE_B_PREFIX) :].strip() or None
        elif upper.startswith(_SEQUENCE_PREFIX):
            sequence = line[len(_SEQUENCE_PREFIX) :].strip() or None
        elif upper.startswith(_SMILES_PREFIX):
            smiles = line[len(_SMILES_PREFIX) :].strip() or None

    # SEQUENCE_B must be checked before SEQUENCE: "SEQUENCE_B: ..." also
    # startswith("SEQUENCE:") above, so order the ifs from longest prefix.
    if not sequence:
        raise ValueError(f"{path} must contain a non-empty 'SEQUENCE:' line")
    if not smiles and not sequence_b:
        raise ValueError(
            f"{path} must contain a non-empty 'SEQUENCE:' line plus a "
            f"'SMILES:' line or a second-protein 'SEQUENCE_B:' line "
            f"(protein+ligand or dimer query)"
        )
    return CofoldInput(sequence=sequence, smiles=smiles or "", sequence_b=sequence_b or "")


def parse_sequence_smiles_file(path: str | Path) -> tuple[str, str]:
    """Read a SEQUENCE:/SMILES: text file, return (sequence, smiles).

    Kept for the protein+ligand CLIs (RF3, ESMFold), which don't take a
    second sequence; unlike parse_input_file it rejects files with no
    SMILES line.
    """
    parsed = parse_input_file(path)
    if not parsed.smiles:
        raise ValueError(
            f"{path} must contain a non-empty 'SEQUENCE:' line and 'SMILES:' line"
        )
    return parsed.sequence, parsed.smiles