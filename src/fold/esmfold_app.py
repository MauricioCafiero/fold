"""Modal app: ESMFold single-sequence structure prediction.

ESMFold's backbone is the 3B-parameter ESM2 model (~11GB of fp32 weights) --
too large to load comfortably on a 14GB-RAM CPU-only laptop alongside
everything else, so (like OpenFold3 cofolding) this runs on a Modal GPU
worker instead of locally.

No MSA search needed (that's the point of ESMFold vs AlphaFold-style
models), so unlike the cofolding app this needs no big preprocessing
pipeline -- just the HF checkpoint, cached in a Modal Volume so the
multi-GB download only happens once.
"""

from pathlib import Path

import modal

APP_NAME = "esmfold"
HF_CACHE_DIR = "/hf-cache"
MODEL_NAME = "facebook/esmfold_v1"

app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("esmfold-hf-cache", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers", "accelerate"
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume},
    timeout=20 * 60,
)
def fold_sequence(sequence: str) -> bytes:
    import os

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    os.environ["HF_HOME"] = HF_CACHE_DIR

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    model = model.cuda()
    model.eval()

    tokens = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    tokens = {k: v.cuda() for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)

    pdb_str = model.output_to_pdb(output)[0]

    hf_cache_volume.commit()
    return pdb_str.encode()


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
