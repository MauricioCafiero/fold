"""Locate the best-ranked structure in a fold job's output directory."""

import json
from pathlib import Path


def find_best_cif(job_dir: str | Path) -> Path:
    """Return the highest-ranked model.cif from an OpenFold3 or RF3 output dir.

    RF3 already promotes its best sample to an unsuffixed <job_name>_model.cif
    at the top of job_dir -- used directly if present. OpenFold3 writes one
    model.cif per sample with no aggregate file, so every sample is compared
    by its confidence file's sample_ranking_score / ranking_score field.
    """
    job_dir = Path(job_dir)

    top_level = list(job_dir.glob("*_model.cif"))
    if top_level:
        return top_level[0]

    best_cif, best_score = None, float("-inf")
    for cif_path in job_dir.rglob("*_model.cif"):
        stem = cif_path.stem
        if stem.endswith("_model"):
            stem = stem[: -len("_model")]

        conf_path = cif_path.with_name(f"{stem}_confidences_aggregated.json")
        if not conf_path.exists():
            conf_path = cif_path.with_name(f"{stem}_summary_confidences.json")
        if not conf_path.exists():
            continue

        data = json.loads(conf_path.read_text())
        score = data.get("sample_ranking_score", data.get("ranking_score"))
        if score is not None and score > best_score:
            best_score = score
            best_cif = cif_path

    if best_cif is None:
        raise FileNotFoundError(
            f"no *_model.cif with a matching confidence file found under {job_dir}"
        )
    return best_cif
