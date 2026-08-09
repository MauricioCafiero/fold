"""Local ESM2 protein embeddings + cosine similarity.

Unlike OpenFold3 cofolding and ESMFold, plain ESM2 is small enough to run
comfortably on CPU: even the 650M-parameter variant is ~2.6GB of weights,
well inside this machine's ~14GB RAM. So this runs locally, no Modal.

Weights come from the HuggingFace Hub via `from_pretrained`, which already
caches to disk on first download (default: ~/.cache/huggingface/hub, or
$HF_HOME if set) and reads from that cache on every later call -- no custom
caching logic needed here, just don't bypass it.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

_model_cache: dict[str, tuple[AutoTokenizer, AutoModel]] = {}


def _load(model_name: str) -> tuple[AutoTokenizer, AutoModel]:
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


@torch.inference_mode()
def embed_sequence(sequence: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Mean-pooled per-residue ESM2 embedding for one protein sequence."""
    tokenizer, model = _load(model_name)
    tokens = tokenizer(sequence, return_tensors="pt")
    hidden = model(**tokens).last_hidden_state[0]  # (seq_len+2, hidden_dim)
    residue_hidden = hidden[1:-1]  # drop BOS/EOS special tokens
    return residue_hidden.mean(dim=0).numpy()


def cosine_similarity(seq_a: str, seq_b: str, model_name: str = DEFAULT_MODEL) -> float:
    """Cosine similarity between two proteins' mean-pooled ESM2 embeddings."""
    a = embed_sequence(seq_a, model_name)
    b = embed_sequence(seq_b, model_name)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    from fold.targets import HMGCR_1HWL_SEQUENCE

    decoy = HMGCR_1HWL_SEQUENCE[:200] + HMGCR_1HWL_SEQUENCE[200:][::-1]
    sim_self = cosine_similarity(HMGCR_1HWL_SEQUENCE, HMGCR_1HWL_SEQUENCE)
    sim_decoy = cosine_similarity(HMGCR_1HWL_SEQUENCE, decoy)
    print(f"self similarity:  {sim_self:.4f}")
    print(f"decoy similarity: {sim_decoy:.4f}")
