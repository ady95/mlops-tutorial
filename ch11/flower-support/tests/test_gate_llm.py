"""LLM 배포 게이트 판정 단위 테스트 (09-5). MLflow·GPU 없이 돈다."""

from src.gate import SLICES, judge

GATE = {"min_json_valid": 1.0, "min_all_fields": 0.80, "min_slice_all_fields": 0.60, "min_summary_pass": 0.80,
        "min_indist_all_fields": 0.97, "max_drop_vs_champion": 0.02}


def metrics(all_fields=0.9, slice_value=0.9, summary=0.9, json_valid=1.0, **slices):
    m = {"eval_json_valid": json_valid, "eval_all_fields": all_fields, "judge_summary_pass": summary}
    m |= {f"slice_{s}": slices.get(s, slice_value) for s in SLICES}
    return m


INDIST = {"indist_all_fields": 0.99}


def test_good_candidate_passes():
    assert all(ok for _, ok in judge(metrics(), INDIST, None, GATE))


def test_one_weak_slice_blocks_even_if_average_is_fine():
    checks = judge(metrics(all_fields=0.85, injection=0.30), INDIST, None, GATE)
    assert not all(ok for _, ok in checks)
    assert [text for text, ok in checks if not ok] == ["  조각 injection 0.30 >= 0.6"]


def test_single_broken_json_blocks():
    assert not all(ok for _, ok in judge(metrics(json_valid=0.99), INDIST, None, GATE))


def test_regression_on_training_distribution_blocks():
    assert not all(ok for _, ok in judge(metrics(), {"indist_all_fields": 0.95}, None, GATE))


def test_worse_than_champion_blocks():
    checks = judge(metrics(all_fields=0.85), INDIST, {"eval_all_fields": "0.9000"}, GATE)
    assert checks[-1][1] is False
