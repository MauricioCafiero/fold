"""Modal app: run RosettaFold3 (RF3) protein/ligand cofolding on Modal GPUs.

A second, architecturally-independent cofolding method alongside OpenFold3
(app.py) -- useful for cross-checking predictions, and RF3 specifically
does better on ligand chirality in published benchmarks. Same pattern as
app.py: pull the vendor image, cache one model checkpoint in a Modal
Volume on first use, run the CLI, return output files.

Reference: https://github.com/RosettaCommons/foundry/blob/production/models/rf3/README.md
"""

import json
import os
from pathlib import Path

import modal

APP_NAME = "rf3-cofold"
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"

# "Latest" (recommended) checkpoint per the RF3 README -- deliberately not
# the `rosettacommons/foundry:weights` tag, which bundles ~18GB covering
# multiple foundry models, not just this one checkpoint.
CKPT_URL = "http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt"
CKPT_FILENAME = "rf3_foundry_01_24_latest_remapped.ckpt"

app = modal.App(APP_NAME)

weights_volume = modal.Volume.from_name("rf3-weights", create_if_missing=True)
outputs_volume = modal.Volume.from_name("rf3-outputs", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("rc-foundry[rf3]")


def _ensure_weights() -> None:
    import urllib.request

    dst = Path(CACHE_DIR) / CKPT_FILENAME
    if dst.exists() and dst.stat().st_size > 0:
        print(f"weights already cached at {dst} ({dst.stat().st_size} bytes)")
        return

    print(f"downloading weights -> {dst}")
    # Plain urllib, not curl/wget -- avoids depending on OS binaries that may
    # be missing from a slim base image (learned the hard way with the
    # OpenFold3 image).
    urllib.request.urlretrieve(CKPT_URL, dst)
    weights_volume.commit()


@app.function(
    image=image,
    volumes={CACHE_DIR: weights_volume, OUTPUT_DIR: outputs_volume},
    timeout=30 * 60,
)
def predict(query_components: list[dict], job_name: str) -> dict[str, bytes]:
    """Run one RF3 cofolding query and return output files as bytes."""
    import subprocess

    _ensure_weights()

    input_json = [{"name": job_name, "components": query_components}]
    input_path = Path("/tmp") / f"{job_name}_input.json"
    input_path.write_text(json.dumps(input_json, indent=2))

    job_out = Path(OUTPUT_DIR) / job_name
    job_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rf3",
        "fold",
        f"inputs={input_path}",
        f"out_dir={job_out}",
        f"ckpt_path={CACHE_DIR}/{CKPT_FILENAME}",
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
    gpu: str = "A10G",
    job_name: str = "hmgcr_rosuvastatin_smoketest",
):
    if not sequence or not smiles:
        from fold.targets import HMGCR_1HWL_SEQUENCE, ROSUVASTATIN_SMILES

        sequence = sequence or HMGCR_1HWL_SEQUENCE
        smiles = smiles or ROSUVASTATIN_SMILES
        print("no --sequence/--smiles given, using HMGCR + rosuvastatin as a smoke test")

    query_components = [
        {"seq": sequence, "chain_id": "A"},
        {"smiles": smiles},
    ]

    fn = predict.with_options(gpu=gpu)
    print(f"submitting {job_name} on {gpu} ...")
    files = fn.remote(query_components, job_name)

    out_dir = Path("outputs") / f"rf3_{job_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, data in files.items():
        dest = out_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print("wrote", dest)
