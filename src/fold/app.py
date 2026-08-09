"""Modal app: run OpenFold3 protein/ligand cofolding on Modal GPUs.

The actual logic (weight caching, query construction, invoking the
`run_openfold` CLI) lives in openfold3_core.py, shared with the no-Modal
runner (local_run.py) for people without a Modal account -- see the Colab
notebook at notebooks/fold_colab.ipynb. This module is just the Modal
plumbing: image, GPU, volumes for persistent caching, and the CLI
entrypoint.
"""

from pathlib import Path

import modal

APP_NAME = "openfold3-cofold"
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"

app = modal.App(APP_NAME)

weights_volume = modal.Volume.from_name("openfold3-weights", create_if_missing=True)
outputs_volume = modal.Volume.from_name("openfold3-outputs", create_if_missing=True)

image = modal.Image.from_registry(
    "openfoldconsortium/openfold3:stable", add_python="3.12"
).pip_install(
    # The published image itself ships a typing_extensions too old for its own
    # pinned pydantic-core (deepspeed -> pydantic -> pydantic_core needs
    # `Sentinel`, added in typing_extensions 4.13). The image already has
    # 4.12.2, which satisfies a >=4.10 bound without upgrading -- pin higher
    # to force the actual bump. Only this one package, to avoid re-resolving
    # anything else in the curated env.
    "typing_extensions>=4.13",
).add_local_python_source("fold")


@app.function(
    image=image,
    volumes={CACHE_DIR: weights_volume, OUTPUT_DIR: outputs_volume},
    timeout=30 * 60,
)
def predict(sequence: str, smiles: str, job_name: str) -> dict[str, bytes]:
    """Run one OpenFold3 cofolding query and return output files as bytes."""
    from fold.openfold3_core import run_predict

    job_out = run_predict(
        sequence, smiles, job_name, Path(CACHE_DIR), Path(OUTPUT_DIR)
    )
    weights_volume.commit()
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

    fn = predict.with_options(gpu=gpu)
    print(f"submitting {job_name} on {gpu} ...")
    files = fn.remote(sequence, smiles, job_name)

    out_dir = Path("outputs") / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, data in files.items():
        dest = out_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print("wrote", dest)
