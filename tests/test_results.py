import json

import pytest

from fold.results import find_best_cif


def test_prefers_top_level_aggregate_cif(tmp_path):
    (tmp_path / "job_model.cif").write_text("data_structure\n")
    (tmp_path / "seed-0_sample-0").mkdir()
    (tmp_path / "seed-0_sample-0" / "job_seed-0_sample-0_model.cif").write_text("data_structure\n")

    result = find_best_cif(tmp_path)

    assert result == tmp_path / "job_model.cif"


def test_picks_highest_ranking_sample_when_no_aggregate(tmp_path):
    for i, score in [(1, 0.80), (2, 0.95), (3, 0.60)]:
        sample_dir = tmp_path / "seed_42"
        sample_dir.mkdir(exist_ok=True)
        stem = f"job_seed_42_sample_{i}"
        (sample_dir / f"{stem}_model.cif").write_text("data_structure\n")
        (sample_dir / f"{stem}_confidences_aggregated.json").write_text(
            json.dumps({"sample_ranking_score": score})
        )

    result = find_best_cif(tmp_path)

    assert result.name == "job_seed_42_sample_2_model.cif"


def test_handles_ranking_score_key_variant(tmp_path):
    stem = "job_seed-0_sample-0"
    (tmp_path / f"{stem}_model.cif").write_text("data_structure\n")
    (tmp_path / f"{stem}_summary_confidences.json").write_text(
        json.dumps({"ranking_score": 0.5})
    )

    result = find_best_cif(tmp_path)

    assert result.name == f"{stem}_model.cif"


def test_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_best_cif(tmp_path)
