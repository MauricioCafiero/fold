"""Core RosettaFold3 (RF3) cofolding logic -- environment-agnostic.

No Modal imports here. Used by the Modal app (rf3_app.py) for remote GPU
execution, and by local_run.py for running directly on any CUDA machine
(e.g. a Colab GPU runtime) that already has `rc-foundry[rf3]` installed.

Includes its own ColabFold MSA-server client: RF3's own CLI has no MSA
search built in (its README says on-the-fly MSA computation is "on the
roadmap", not implemented yet), only a `msa_path` field for a precomputed
`.a3m`/`.fasta`. This replicates the relevant bits of ColabFold's own
`run_mmseqs2` client
(https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py)
for the single-protein-sequence, unpaired, no-templates case needed here.
"""

import json
import subprocess
from pathlib import Path

# "Latest" (recommended) checkpoint per the RF3 README -- deliberately not
# the `rosettacommons/foundry:weights` Docker tag, which bundles ~18GB
# covering multiple foundry models, not just this one checkpoint.
CKPT_URL = "http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt"
CKPT_FILENAME = "rf3_foundry_01_24_latest_remapped.ckpt"

MSA_SERVER_URL = "https://api.colabfold.com"


def fetch_msa(sequence: str, out_path: Path) -> None:
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


def ensure_weights(cache_dir: Path) -> Path:
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / CKPT_FILENAME
    if dst.exists() and dst.stat().st_size > 0:
        print(f"weights already cached at {dst} ({dst.stat().st_size} bytes)")
        return dst

    print(f"downloading RF3 weights -> {dst}")
    # Plain urllib, not curl/wget -- avoids depending on OS binaries that may
    # be missing from a minimal environment.
    urllib.request.urlretrieve(CKPT_URL, dst)
    return dst


def build_query_components(sequence: str, smiles: str) -> list[dict]:
    return [
        {"seq": sequence, "chain_id": "A"},
        {"smiles": smiles},
    ]


def run_predict(
    sequence: str,
    smiles: str,
    job_name: str,
    cache_dir: Path,
    output_dir: Path,
    work_dir: Path = Path("/tmp"),
    use_msa: bool = True,
) -> Path:
    """Run one RF3 cofolding job. Returns the job's output directory.

    Assumes the `rf3` CLI is already installed and importable on PATH in
    the current environment.
    """
    ckpt_path = ensure_weights(cache_dir)
    query_components = build_query_components(sequence, smiles)

    if use_msa:
        for i, component in enumerate(query_components):
            if "seq" in component and "msa_path" not in component:
                msa_path = work_dir / f"{job_name}_chain{i}_msa.a3m"
                print(f"fetching MSA for component {i} ({len(component['seq'])} aa)...")
                fetch_msa(component["seq"], msa_path)
                component["msa_path"] = str(msa_path)

    input_json = [{"name": job_name, "components": query_components}]
    input_path = work_dir / f"{job_name}_input.json"
    input_path.write_text(json.dumps(input_json, indent=2))

    job_out = output_dir / job_name
    job_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rf3",
        "fold",
        f"inputs={input_path}",
        f"out_dir={job_out}",
        f"ckpt_path={ckpt_path}",
    ]

    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-8000:])
    print(result.stderr[-8000:])
    result.check_returncode()

    return job_out
