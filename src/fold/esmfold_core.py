"""Core ESMFold single-sequence structure prediction logic -- environment-agnostic.

No Modal imports here. Used by the Modal app (esmfold_app.py) for remote
GPU execution, and by local_run.py for running directly on any CUDA
machine (e.g. a Colab GPU runtime) that already has
torch/transformers/accelerate installed.
"""

MODEL_NAME = "facebook/esmfold_v1"


def fold_sequence(sequence: str) -> bytes:
    """Run ESMFold on one sequence, return a PDB structure as bytes."""
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    model = model.cuda()
    model.eval()

    tokens = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    tokens = {k: v.cuda() for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)

    return model.output_to_pdb(output)[0].encode()
