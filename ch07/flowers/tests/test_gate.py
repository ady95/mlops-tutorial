"""배포 승인 판정 로직 단위 테스트 (07-4). MLflow 서버 없이 돈다."""

from src.registry import judge

GATE = {"min_test_f1": 0.85, "max_latency_ms": 20, "max_f1_drop": 0.0, "require_clean_git": True}


def tags(f1="0.9259", latency="8.6", dirty="False"):
    return {"test_f1_macro": f1, "test_latency_ms": latency, "git_dirty": dirty}


def test_first_model_passes_without_champion():
    checks = judge(tags(), None, GATE)
    assert all(ok for _, ok in checks)
    assert len(checks) == 3            # champion이 없으면 비교 항목이 없다


def test_dirty_git_is_rejected():      # 5장의 v2
    checks = judge(tags(dirty="True"), None, GATE)
    assert not all(ok for _, ok in checks)


def test_worse_than_champion_is_rejected():   # 5장의 v3
    checks = judge(tags(f1="0.9150"), tags(f1="0.9259"), GATE)
    assert [ok for _, ok in checks] == [True, True, True, False]


def test_slow_model_is_rejected():
    checks = judge(tags(latency="42.9"), None, GATE)
    assert checks[1][1] is False
