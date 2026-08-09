"""Post-processing for fold outputs: CIF->PDB conversion and binding-energy
calculation. CPU-only, no GPU/Modal needed -- separate from local_run.py,
which is specifically the GPU-requiring cofolding/folding tools.

    python -m fold.analyze cif-to-pdb outputs/.../model.cif
    python -m fold.analyze binding-energy outputs/hmgcr_rosuvastatin_smoketest --smiles "..."
"""

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("cif-to-pdb")
def cif_to_pdb_cmd(cif_path: str, out_path: str = ""):
    """Convert a cofolding output .cif to .pdb (e.g. for use with MD tools)."""
    from fold.structure_convert import cif_to_pdb

    dest = cif_to_pdb(cif_path, out_path or None)
    print(f"wrote {dest}")


@app.command("binding-energy")
def binding_energy_cmd(
    job_dir: str,
    smiles: str = "",
    ligand_chain: str = "",
    box_size: float = 22.0,
):
    """Pick the best-ranked structure from a fold job's output dir, convert
    to PDB, and calculate a real AutoDock Vina binding affinity (kcal/mol).

    Requires the 'docking' extra (`uv sync --extra docking`) and the
    `obabel` CLI on PATH.
    """
    from fold.binding_energy import calculate_binding_energy_for_job

    if not smiles:
        from fold.targets import ROSUVASTATIN_SMILES

        smiles = ROSUVASTATIN_SMILES
        print("no --smiles given, using rosuvastatin as a smoke test")

    result = calculate_binding_energy_for_job(
        job_dir, smiles, ligand_chain=ligand_chain or None, box_size=box_size
    )
    print()
    print(f"affinity: {result['affinity_kcal_mol']} kcal/mol")
    print(f"box center: {result['box_center']}")
    print(f"receptor PDB: {result['receptor_pdb']}")
    print(f"pose SDF: {result['pose_sdf']}")


if __name__ == "__main__":
    app()
