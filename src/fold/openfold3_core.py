"""Core OpenFold3 cofolding logic -- environment-agnostic.

No Modal imports here. Used by the Modal app (app.py) for remote GPU
execution, and by local_run.py for running directly on any CUDA machine
(e.g. a Colab GPU runtime) that already has the `run_openfold` CLI
installed. This module only handles weight caching, query construction,
and invoking that CLI -- installing it is the caller's job, since that
differs between a Modal image build and a `pip install` in Colab.
"""

import json
import subprocess
from pathlib import Path

# OpenBind checkpoint that ships with openfold3 0.5.0 (v0.5.0, "OpenBind
# Model Release") -- plain HTTPS mirror of the RODA bucket object from the
# OpenFold3 install docs
# (https://openfold-3.readthedocs.io/en/latest/Installation.html), which
# document `aws s3 cp s3://openfold3-data/openfold3-parameters/of3-ob-2025-06-30-174k.pt ... --no-sign-request`.
# Verified reachable directly over HTTPS (200, Content-Length 2287872989).
# NOTE: this pairs with openfold3 >= 0.5.0 -- the earlier of3-p2-155k.pt
# checkpoint belongs to the 0.4.x architecture (its state_dict layout, e.g.
# per-block attention layer_norm_z, no longer matches 0.5's model).
WEIGHTS_URL = (
    "https://openfold3-data.s3.amazonaws.com/openfold3-parameters/"
    "of3-ob-2025-06-30-174k.pt"
)
WEIGHTS_FILENAME = "of3-ob-2025-06-30-174k.pt"


def ensure_weights(cache_dir: Path) -> Path:
    """Download the OpenFold3 checkpoint into cache_dir if not already there."""
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / WEIGHTS_FILENAME
    if dst.exists() and dst.stat().st_size > 0:
        print(f"weights already cached at {dst} ({dst.stat().st_size} bytes)")
        return dst

    print(f"downloading OpenFold3 weights (~2.1GB) -> {dst}")
    urllib.request.urlretrieve(WEIGHTS_URL, dst)
    return dst


def build_query_chains(
    sequence: str, smiles: str = "", second_sequence: str = ""
) -> list[dict]:
    """Chains for one query: protein (+ optional partner) + optional ligand.

    With `second_sequence` set this is a protein-protein (dimer) query --
    chains A and B, no ligand unless a `smiles` is also given.
    """
    chains = [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": sequence}]
    if second_sequence:
        chains.append(
            {"molecule_type": "protein", "chain_ids": ["B"], "sequence": second_sequence}
        )
    if smiles:
        chains.append({"molecule_type": "ligand", "chain_ids": ["Z"], "smiles": smiles})
    # a single protein chain is a valid apo query; openfold3 folds it alone
    return chains


def build_pocket_constraint(
    pocket_residues: tuple[int, ...] | list[int],
    max_distance: float = 4.0,
    ligand_chain_id: str = "Z",
) -> dict:
    """OpenFold3 `pocket_constraint` biasing the ligand toward given residues.

    Residue positions are 1-based query-sequence positions on chain A, which
    is what the schema wants for a sequence input. Granularity is the residue:
    there is no ligand-atom side, so a Boltz `contact` hint's `token2` has
    nowhere to go. Present since openfold3 0.4.5, so the pinned 0.5.0 image
    takes it as-is.

    This BIASES; it does not enforce. Supplying it makes OpenFold3 rank 1,024
    rigid ligand proposals on pocket proximity, clash and a contact penalty,
    then seed partial-diffusion refinement over the last quarter of the
    schedule -- and if its diversity filter leaves too few poses, the rest are
    filled from the sorted list whatever their contact quality. So a run with
    this set is "pocket-guided", never "forced".
    """
    return {
        "ligand_chain_id": ligand_chain_id,
        "pocket_residues": [["A", int(r)] for r in pocket_residues],
        "max_distance": float(max_distance),
    }


def run_predict(
    sequence: str,
    smiles: str,
    job_name: str,
    cache_dir: Path,
    output_dir: Path,
    second_sequence: str = "",
    work_dir: Path = Path("/tmp"),
    pocket_residues: tuple[int, ...] | list[int] = (),
    pocket_max_distance: float = 4.0,
    seed: int | None = None,
) -> Path:
    """Run one OpenFold3 cofolding job. Returns the job's output directory.

    Assumes the `run_openfold` CLI is already installed and importable on
    PATH in the current environment.

    `pocket_residues` adds a pocket_constraint (see build_pocket_constraint --
    it biases ligand placement, it does not force contacts).

    `seed` overrides OpenFold3's default experiment_settings.seeds = [42]
    via a runner yaml. Unset, every run is seed 42 and a re-run reproduces
    the same fold (to ~0.02 A, measured 2026-11-01), so fold replicates
    need a distinct seed per draw.
    """
    ckpt_path = ensure_weights(cache_dir)

    # MSA flags live on the Query, not per-chain (per openfold3's actual
    # pydantic schema in inference_query_format.py, which differs from the
    # readthedocs example).
    query = {
        "chains": build_query_chains(sequence, smiles, second_sequence),
        "use_msas": True,
        "use_main_msas": True,
        "use_paired_msas": True,
    }
    if pocket_residues:
        if not smiles:
            raise ValueError("pocket_residues needs a ligand (smiles) to constrain")
        # one per query: the schema's field is singular
        query["pocket_constraint"] = build_pocket_constraint(
            pocket_residues, pocket_max_distance
        )
        print(
            f"pocket-guided run: {len(pocket_residues)} residues, "
            f"max_distance {pocket_max_distance} A (biases placement, does not force it)"
        )
    query_path = work_dir / f"{job_name}_query.json"
    query_path.write_text(json.dumps({"queries": {job_name: query}}, indent=2))

    job_out = output_dir / job_name
    job_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "run_openfold",
        "predict",
        f"--query_json={query_path}",
        f"--output_dir={job_out}",
        f"--inference_ckpt_path={ckpt_path}",
        "--use_msa_server=true",
        "--use_templates=false",
    ]
    if seed is not None:
        runner_path = work_dir / f"{job_name}_runner.yml"
        runner_path.write_text(f"experiment_settings:\n  seeds: [{int(seed)}]\n")
        cmd.append(f"--runner_yaml={runner_path}")

    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-8000:])
    print(result.stderr[-8000:])
    result.check_returncode()

    return job_out
