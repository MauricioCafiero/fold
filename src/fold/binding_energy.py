"""Calculate a real AutoDock Vina binding affinity for a cofolded complex.

Docking logic is vina_dock_core.py, vendored from MauricioCafiero/dock_assist
and trimmed to the "dock at a known site" path -- no external sibling-repo
dependency.

Rather than blind-docking (searching the whole receptor for a pocket) or
just measuring geometric proximity, this anchors the Vina search box at
the COFOLDED LIGAND'S OWN PREDICTED POSITION -- the same "docking at a
known site" path dock_assist uses for redocking against a co-crystallized
ligand (see vina_dock_core.dock_at_centroid). The reported affinity is
therefore a real, calculated binding energy (kcal/mol) for a search
centered on where OpenFold3/RF3 actually placed the ligand, not wherever a
blind pocket scan happens to find.

Needs its own optional dependencies -- see pyproject.toml's
[project.optional-dependencies] "docking" group; run
`uv sync --extra docking` before using this module.
"""

import re
from pathlib import Path

import numpy as np


def detect_ligand_chain(pdb_path: Path) -> str:
    """Return the chain ID carrying HETATM records, i.e. the ligand.

    OpenFold3 and RF3 don't agree on a chain letter for the ligand (this
    repo's queries put it on "Z" for OpenFold3, but RF3 auto-assigns its own
    -- e.g. "B" -- since its query schema doesn't request one). Auto-detecting
    avoids hardcoding either convention. Raises if zero or more than one
    distinct HETATM chain is found (single-ligand cofolds only).
    """
    chains = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("HETATM"):
                chains.add(line[21])
    if not chains:
        raise ValueError(f"no HETATM (ligand) records found in {pdb_path}")
    if len(chains) > 1:
        raise ValueError(
            f"expected exactly one ligand chain in {pdb_path}, found {sorted(chains)} "
            "-- pass ligand_chain explicitly."
        )
    return chains.pop()


def split_receptor_and_ligand_centroid(
    pdb_path: Path, ligand_chain: str, receptor_pdb_path: Path
) -> tuple[Path, tuple[float, float, float]]:
    """Write a receptor-only PDB (ligand chain's atoms dropped) and return
    the ligand's centroid from its own predicted coordinates.

    Column offsets match dock_assist's own PDB parsing (vina_dock.py's
    parse_pdb_heavy_atoms / _crystal_ligand_coords), verified against a
    real converted OpenFold3 output.
    """
    ligand_coords = []
    receptor_lines = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                if line[21] == ligand_chain:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ligand_coords.append((x, y, z))
                    continue  # drop from the receptor-only file
            receptor_lines.append(line)

    if not ligand_coords:
        raise ValueError(f"no atoms found for ligand chain {ligand_chain!r} in {pdb_path}")

    receptor_pdb_path.write_text("".join(receptor_lines))
    x, y, z = np.mean(ligand_coords, axis=0)
    return receptor_pdb_path, (float(x), float(y), float(z))


def calculate_binding_energy(
    structure_path: str | Path,
    smiles: str,
    ligand_chain: str | None = None,
    box_size: float = 22.0,
) -> dict:
    """Compute a real AutoDock Vina binding affinity (kcal/mol) for one
    cofolded protein-ligand complex.

    Accepts a .cif (converted to PDB first) or .pdb structure. ligand_chain
    is auto-detected from the HETATM records if not given (see
    detect_ligand_chain) -- OpenFold3 and RF3 don't use the same chain
    letter for the ligand. Returns a dict: affinity_kcal_mol (float or None
    if Vina failed to score), the box center used, the receptor-only PDB
    path, the docked-pose SDF path (parsed from the docking report), and
    the full report text.
    """
    from fold import vina_dock_core

    structure_path = Path(structure_path)
    if structure_path.suffix.lower() == ".cif":
        from fold.structure_convert import cif_to_pdb

        pdb_path = cif_to_pdb(structure_path)
    else:
        pdb_path = structure_path

    if ligand_chain is None:
        ligand_chain = detect_ligand_chain(pdb_path)
        print(f"auto-detected ligand chain: {ligand_chain!r}")

    receptor_pdb_path = pdb_path.with_name(pdb_path.stem + "_receptor.pdb")
    receptor_pdb_path, centroid = split_receptor_and_ligand_centroid(
        pdb_path, ligand_chain, receptor_pdb_path
    )

    report = vina_dock_core.dock_at_centroid(
        str(receptor_pdb_path), [smiles], center=centroid, box_size=box_size
    )

    affinity_match = re.search(r"score:\s*(-?\d+\.\d+)\s*kcal/mol", report)
    sdf_match = re.search(r"poses SDF.*?:\s*(\S+)", report)

    return {
        "affinity_kcal_mol": float(affinity_match.group(1)) if affinity_match else None,
        "box_center": centroid,
        "receptor_pdb": str(receptor_pdb_path),
        "pose_sdf": sdf_match.group(1) if sdf_match else None,
        "report": report,
    }


def calculate_binding_energy_for_job(
    job_dir: str | Path,
    smiles: str,
    ligand_chain: str | None = None,
    box_size: float = 22.0,
) -> dict:
    """Full workflow: pick the best-ranked structure from a fold job's output
    directory, convert it to PDB, and calculate its Vina binding energy.

    job_dir is an OpenFold3/RF3 output directory as written by app.py /
    rf3_app.py / local_run.py, e.g. outputs/<job_name> or
    outputs/rf3_<job_name>. See results.find_best_cif for how "best" is
    chosen.
    """
    from fold.results import find_best_cif

    best_cif = find_best_cif(job_dir)
    print(f"best-ranked structure: {best_cif}")
    return calculate_binding_energy(best_cif, smiles, ligand_chain, box_size)
