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

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "rc-foundry[rf3]", "requests"
)

# Public ColabFold MSA server -- same one OpenFold3's --use_msa_server hits
# internally. RF3's own CLI has no equivalent auto-search (its README says
# on-the-fly MSA computation is "on the roadmap", not implemented yet), so
# this replicates the relevant bits of ColabFold's own `run_mmseqs2` client
# (https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py)
# for the single-protein-sequence, unpaired, no-templates case we need.
MSA_SERVER_URL = "https://api.colabfold.com"


def _fetch_msa(sequence: str, out_path: Path) -> None:
    import random
    import tarfile
    import time

    import requests

    headers = {"User-Agent": "fold-repo/0.1 (github.com/MauricioCafiero/fold)"}
    query = f">101\n{sequence}\n"

    def submit() -> dict:
        res = requests.post(
            f"{MSA_SERVER_URL}/ticket/msa",
            data={"q": query, "mode": "env"},
            timeout=30,
            headers=headers,
        )
        try:
            return res.json()
        except ValueError:
            raise RuntimeError(f"MSA server returned non-JSON: {res.text}")

    out = submit()
    while out.get("status") in ("UNKNOWN", "RATELIMIT"):
        time.sleep(5 + random.randint(0, 5))
        out = submit()
    if out.get("status") in ("ERROR", "MAINTENANCE"):
        raise RuntimeError(f"ColabFold MSA server error: {out}")

    job_id = out["id"]
    print(f"MSA job {job_id} submitted, polling...")
    while out.get("status") in ("UNKNOWN", "RUNNING", "PENDING"):
        time.sleep(5 + random.randint(0, 5))
        out = requests.get(
            f"{MSA_SERVER_URL}/ticket/{job_id}", timeout=30, headers=headers
        ).json()
    if out.get("status") != "COMPLETE":
        raise RuntimeError(f"ColabFold MSA server job did not complete: {out}")

    res = requests.get(
        f"{MSA_SERVER_URL}/result/download/{job_id}", timeout=60, headers=headers
    )
    tar_path = out_path.parent / f"{job_id}.tar.gz"
    tar_path.write_bytes(res.content)

    extract_dir = out_path.parent / f"{job_id}_extracted"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir)

    # "env" mode returns uniref hits plus environmental (BFD/MGnify/etc.)
    # hits as separate files; concatenating both is the standard unpaired
    # MSA ColabFold-based pipelines feed to structure models.
    combined = ""
    for name in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
        f = extract_dir / name
        if f.exists():
            combined += f.read_text()
    out_path.write_text(combined)
    print(f"MSA written to {out_path} ({len(combined)} chars)")


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
def predict(query_components: list[dict], job_name: str, use_msa: bool = True) -> dict[str, bytes]:
    """Run one RF3 cofolding query and return output files as bytes."""
    import subprocess

    _ensure_weights()

    if use_msa:
        for i, component in enumerate(query_components):
            if "seq" in component and "msa_path" not in component:
                msa_path = Path("/tmp") / f"{job_name}_chain{i}_msa.a3m"
                print(f"fetching MSA for component {i} ({len(component['seq'])} aa)...")
                _fetch_msa(component["seq"], msa_path)
                component["msa_path"] = str(msa_path)

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
    input_file: str = "",
    gpu: str = "A10G",
    job_name: str = "",
    use_msa: bool = True,
):
    if input_file:
        from fold.inputs import parse_sequence_smiles_file

        sequence, smiles = parse_sequence_smiles_file(input_file)
        job_name = job_name or Path(input_file).stem

    job_name = job_name or "hmgcr_rosuvastatin_smoketest"

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
    print(f"submitting {job_name} on {gpu} (use_msa={use_msa}) ...")
    files = fn.remote(query_components, job_name, use_msa)

    out_dir = Path("outputs") / f"rf3_{job_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, data in files.items():
        dest = out_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print("wrote", dest)
