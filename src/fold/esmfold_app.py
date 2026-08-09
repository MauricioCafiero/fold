"""Modal app: ESMFold single-sequence structure prediction.

ESMFold's backbone is the 3B-parameter ESM2 model (~11GB of fp32 weights) --
too large to load comfortably on most CPU-only laptops, so this runs on a
Modal GPU worker instead of locally (or on any CUDA machine directly via
local_run.py / the Colab notebook, sharing esmfold_core.py). No MSA search
needed (that's the point of ESMFold vs AlphaFold-style models), so unlike
the cofolding apps this needs no big preprocessing pipeline -- just the HF
checkpoint, cached in a Modal Volume so the multi-GB download only happens
once.
"""

from pathlib import Path

import modal

APP_NAME = "esmfold"
HF_CACHE_DIR = "/hf-cache"

app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("esmfold-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate")
    .add_local_python_source("fold")
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume},
    timeout=20 * 60,
)
def fold_sequence(sequence: str) -> bytes:
    import os

    from fold.esmfold_core import fold_sequence as _fold_sequence

    os.environ["HF_HOME"] = HF_CACHE_DIR
    pdb_bytes = _fold_sequence(sequence)
    hf_cache_volume.commit()
    return pdb_bytes


@app.local_entrypoint()
def main(sequence: str = "", out_path: str = "outputs/esmfold_prediction.pdb"):
    if not sequence:
        from fold.targets import HMGCR_1HWL_SEQUENCE

        sequence = HMGCR_1HWL_SEQUENCE
        print("no --sequence given, using HMGCR 1HWL construct as a smoke test")

    print(f"folding {len(sequence)}-residue sequence on Modal (A10G) ...")
    pdb_bytes = fold_sequence.remote(sequence)

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdb_bytes)
    print("wrote", dest)
