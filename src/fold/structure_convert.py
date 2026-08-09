"""Convert cofolding output (mmCIF) to PDB, e.g. for use with MD tools that
expect PDB input (GROMACS, OpenMM, AMBER, ...) or with dock_assist (see
binding_energy.py), which works on plain PDB files.
"""

from pathlib import Path

import gemmi


def cif_to_pdb(cif_path: str | Path, pdb_path: str | Path | None = None) -> Path:
    """Convert one mmCIF structure file to PDB. Returns the output path."""
    cif_path = Path(cif_path)
    pdb_path = Path(pdb_path) if pdb_path else cif_path.with_suffix(".pdb")

    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    structure.write_pdb(str(pdb_path))
    return pdb_path
