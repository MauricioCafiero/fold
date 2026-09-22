"""Post-processing for fold outputs: CIF->PDB conversion and binding-energy
calculation. CPU-only, no GPU/Modal needed -- separate from local_run.py,
which is specifically the GPU-requiring cofolding/folding tools.

    python -m fold.analyze cif-to-pdb outputs/.../model.cif
    python -m fold.analyze binding-energy outputs/hmgcr_rosuvastatin_smoketest --smiles "..."
"""

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _print_score_pairs(scores: dict) -> None:
    for chain_1, chain_2 in sorted({tuple(sorted(k.split("-"))) for k in scores["ipsae"]}):
        ipsae = max(scores["ipsae"].get(f"{chain_1}-{chain_2}", 0.0),
                    scores["ipsae"].get(f"{chain_2}-{chain_1}", 0.0))
        pdockq2 = max(scores["pdockq2"].get(f"{chain_1}-{chain_2}", 0.0),
                      scores["pdockq2"].get(f"{chain_2}-{chain_1}", 0.0))
        line = f"  interface {chain_1}-{chain_2}: ipSAE {ipsae:.3f}, pDockQ2 {pdockq2:.3f}"
        if f"{chain_1}-{chain_2}" in scores["pdockq"]:
            line += f", pDockQ {scores['pdockq'][f'{chain_1}-{chain_2}']:.3f}"
        print(line)


def report_interface_scores(job_dir: str | Path, pae_cutoff: float = 15.0) -> None:
    """Print interface confidence for a cofold job's best-ranked model.

    Best-effort summary meant to run right after a cofold: any failure (no
    PAE matrix in the confidences, unsupported chain composition, missing
    files) prints a note instead of failing -- a job that produced
    structures shouldn't error just because its confidences can't be
    scored.
    """
    from fold.ipsae import interface_scores, locate_conf_json
    from fold.results import find_best_cif

    try:
        cif = find_best_cif(job_dir)
        scores = interface_scores(locate_conf_json(cif), cif, pae_cutoff=pae_cutoff)
    except Exception as exc:  # best-effort: report, don't raise
        print(f"interface scores unavailable ({exc})")
        return

    print(f"interface confidence (best model: {cif.name}):")
    _print_score_pairs(scores)


@app.command("cif-to-pdb")
def cif_to_pdb_cmd(cif_path: str, out_path: str = ""):
    """Convert a cofolding output .cif to .pdb (e.g. for use with MD tools)."""
    from fold.structure_convert import cif_to_pdb

    dest = cif_to_pdb(cif_path, out_path or None)
    print(f"wrote {dest}")


@app.command("interface-scores")
def interface_scores_cmd(job_dir: str, pae_cutoff: float = 15.0):
    """Report per-interface ipSAE / pDockQ / pDockQ2 for a cofolding job.

    ipSAE (Dunbrack 2025) is an interface-local confidence that fixes iptm's
    whole-chain bias: it scores only the residue pairs that actually touch.
    Computed from the best-ranked model's full confidences JSON (PAE matrix)
    + CIF. Protein-protein interfaces follow the published benchmark;
    protein-LIGAND interfaces use the same math but are not benchmarked by
    the paper (the ligand is a tiny pseudo-chain, which deflates d0) --
    treat those as experimental. Requires a PAE matrix: RF3 outputs have
    one; OpenFold3 needs an image that includes openfold3 PR #142.
    """
    from fold.ipsae import interface_scores, locate_conf_json
    from fold.results import find_best_cif

    cif = find_best_cif(job_dir)
    scores = interface_scores(locate_conf_json(cif), cif, pae_cutoff=pae_cutoff)
    print(f"best-ranked structure: {cif}")

    _print_score_pairs(scores)


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
