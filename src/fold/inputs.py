"""Parse a plain-text cofolding input file for the CLIs.

Expected format, one field per line, order doesn't matter:

    SEQUENCE: MLSRLFRMHGLFVASHPWEVIVG...
    SMILES: CC(C)C1=NC(=NC(=C1/C=C/[C@H]...
    SEQUENCE_B: TETSSHKAHTEAQVINTFDGV...   (optional, dimer partner)
    POCKET: 4,24,29,34                     (optional, OpenFold3 only)
    POCKET_MAX_DISTANCE: 4.2               (optional, default 4.0)

SEQUENCE plus SMILES is a protein+ligand query; SEQUENCE plus SEQUENCE_B
is a protein+protein (dimer) query; all three cofold two proteins with a
ligand. A SMILES line may be omitted when SEQUENCE_B is present.

POCKET lists 1-based chain-A residue positions to bias the ligand toward,
and becomes OpenFold3's `pocket_constraint`. It is NOT the equivalent of a
Boltz `contact ... force: true` hint: the constraint is residue-level, so
the hint's ligand atom is dropped, and OpenFold3 uses it to rank rigid
ligand proposals seeded into the last quarter of the diffusion schedule
rather than to enforce anything -- nothing rejects a pose that misses the
pocket. Call such runs pocket-guided, not forced. RF3 has no equivalent
input, so POCKET does not apply to that lane; see peptidebuilder's
NEXT_STEPS.md, "Forced vs unforced co-folds".
"""

from dataclasses import dataclass
from pathlib import Path

_SEQUENCE_PREFIX = "SEQUENCE:"
_SEQUENCE_B_PREFIX = "SEQUENCE_B:"
_SMILES_PREFIX = "SMILES:"
_POCKET_PREFIX = "POCKET:"
_POCKET_MAX_DISTANCE_PREFIX = "POCKET_MAX_DISTANCE:"


@dataclass
class CofoldInput:
    """Chains parsed from an input file (subset of the given lines)."""

    sequence: str
    smiles: str = ""
    sequence_b: str = ""
    pocket_residues: tuple[int, ...] = ()
    pocket_max_distance: float = 4.0

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
    pocket: str | None = None
    pocket_max_distance: str | None = None

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
        # POCKET_MAX_DISTANCE before POCKET, same longest-prefix-first rule
        # as SEQUENCE_B above.
        elif upper.startswith(_POCKET_MAX_DISTANCE_PREFIX):
            pocket_max_distance = line[len(_POCKET_MAX_DISTANCE_PREFIX) :].strip() or None
        elif upper.startswith(_POCKET_PREFIX):
            pocket = line[len(_POCKET_PREFIX) :].strip() or None

    # SEQUENCE_B must be checked before SEQUENCE: "SEQUENCE_B: ..." also
    # startswith("SEQUENCE:") above, so order the ifs from longest prefix.
    if not sequence:
        raise ValueError(f"{path} must contain a non-empty 'SEQUENCE:' line")

    residues: tuple[int, ...] = ()
    if pocket:
        residues = tuple(int(tok) for tok in pocket.replace(",", " ").split())
        if not residues:
            raise ValueError(f"{path}: 'POCKET:' line lists no residues")
        if min(residues) < 1 or max(residues) > len(sequence):
            raise ValueError(
                f"{path}: POCKET residues {residues} out of range for a "
                f"{len(sequence)}-residue sequence (positions are 1-based)"
            )
        if not smiles:
            raise ValueError(f"{path}: 'POCKET:' needs a 'SMILES:' ligand to constrain")

    # A bare SEQUENCE: line is a protein-only (apo) query: openfold3 accepts a
    # single chain. SMILES adds a ligand, SEQUENCE_B a second protein chain.
    return CofoldInput(
        sequence=sequence,
        smiles=smiles or "",
        sequence_b=sequence_b or "",
        pocket_residues=residues,
        pocket_max_distance=float(pocket_max_distance) if pocket_max_distance else 4.0,
    )


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