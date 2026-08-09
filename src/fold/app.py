"""Modal app: run OpenFold3 protein/ligand cofolding on Modal GPUs.

The laptop never runs the model itself (no CUDA locally). This module defines
a Modal Function that pulls the official OpenFold3 container image, fetches
model weights into a persistent Modal Volume on first use, and runs
`run_openfold predict` on a GPU worker. The local entrypoint just builds the
query JSON and triggers the remote call.
"""

import json
import os
from pathlib import Path

import modal

APP_NAME = "openfold3-cofold"
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"

# Public, unauthenticated checkpoint per OpenFold3 install docs:
# https://openfold-3.readthedocs.io/en/latest/Installation.html
WEIGHTS_S3_URI = "s3://openfold/staging/of3-p2-155k.pt"
WEIGHTS_FILENAME = "of3-p2-155k.pt"

app = modal.App(APP_NAME)

weights_volume = modal.Volume.from_name("openfold3-weights", create_if_missing=True)
outputs_volume = modal.Volume.from_name("openfold3-outputs", create_if_missing=True)

image = modal.Image.from_registry(
    "openfoldconsortium/openfold3:stable", add_python="3.12"
).run_commands(
    # Standalone AWS CLI v2 bundle (ships its own embedded Python) -- deliberately
    # NOT `pip_install("awscli")`, which re-resolves typing_extensions/pydantic-core
    # inside the image's curated conda env and breaks deepspeed's imports.
    "apt-get update && apt-get install -y --no-install-recommends curl unzip"
    " && curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip"
    " && unzip -q /tmp/awscliv2.zip -d /tmp"
    " && /tmp/aws/install"
    " && rm -rf /tmp/awscliv2.zip /tmp/aws /var/lib/apt/lists/*"
).pip_install(
    # The published image itself ships a typing_extensions too old for its own
    # pinned pydantic-core (deepspeed -> pydantic -> pydantic_core needs
    # `Sentinel`, added in typing_extensions 4.13). The image already has
    # 4.12.2, which satisfies a >=4.10 bound without upgrading -- pin higher
    # to force the actual bump. Only this one package, to avoid re-resolving
    # anything else in the curated env.
    "typing_extensions>=4.13",
)


def _ensure_weights() -> None:
    import subprocess

    dst = Path(CACHE_DIR) / WEIGHTS_FILENAME
    if dst.exists() and dst.stat().st_size > 0:
        print(f"weights already cached at {dst} ({dst.stat().st_size} bytes)")
        return

    print(f"downloading weights -> {dst}")
    subprocess.run(
        ["aws", "s3", "cp", WEIGHTS_S3_URI, str(dst), "--no-sign-request"],
        check=True,
    )
    weights_volume.commit()


@app.function(
    image=image,
    volumes={CACHE_DIR: weights_volume, OUTPUT_DIR: outputs_volume},
    timeout=30 * 60,
)
def predict(query_chains: list[dict], job_name: str) -> dict[str, bytes]:
    """Run one OpenFold3 cofolding query and return output files as bytes."""
    import subprocess

    os.environ["OPENFOLD_CACHE"] = CACHE_DIR
    _ensure_weights()

    # MSA flags live on the Query, not per-chain (per openfold3's actual
    # pydantic schema in inference_query_format.py, which differs from the
    # readthedocs example).
    query = {
        "chains": query_chains,
        "use_msas": True,
        "use_main_msas": True,
        "use_paired_msas": True,
    }
    query_path = Path("/tmp") / f"{job_name}_query.json"
    query_path.write_text(json.dumps({"queries": {job_name: query}}, indent=2))

    job_out = Path(OUTPUT_DIR) / job_name
    job_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "run_openfold",
        "predict",
        f"--query_json={query_path}",
        f"--output_dir={job_out}",
        f"--inference_ckpt_path={CACHE_DIR}/{WEIGHTS_FILENAME}",
        "--use_msa_server=true",
        "--use_templates=false",
    ]

    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-8000:])
    print(result.stderr[-8000:])
    result.check_returncode()

    outputs_volume.commit()

    return {
        str(p.relative_to(job_out)): p.read_bytes()
        for p in job_out.rglob("*")
        if p.is_file()
    }


@app.local_entrypoint()
def main(
    sequence: str = "",
    smiles: str = "",
    input_file: str = "",
    gpu: str = "A10G",
    job_name: str = "",
):
    if input_file:
        from fold.inputs import parse_sequence_smiles_file

        sequence, smiles = parse_sequence_smiles_file(input_file)
        job_name = job_name or Path(input_file).stem

    job_name = job_name or "prediction"

    if not sequence or not smiles:
        from fold.targets import HMGCR_1HWL_SEQUENCE, ROSUVASTATIN_SMILES

        sequence = sequence or HMGCR_1HWL_SEQUENCE
        smiles = smiles or ROSUVASTATIN_SMILES
        print("no --sequence/--smiles given, using HMGCR + rosuvastatin as a smoke test")

    query_chains = [
        {
            "molecule_type": "protein",
            "chain_ids": ["A"],
            "sequence": sequence,
        },
        {
            "molecule_type": "ligand",
            "chain_ids": ["Z"],
            "smiles": smiles,
        },
    ]

    fn = predict.with_options(gpu=gpu)
    print(f"submitting {job_name} on {gpu} ...")
    files = fn.remote(query_chains, job_name)

    out_dir = Path("outputs") / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, data in files.items():
        dest = out_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print("wrote", dest)
