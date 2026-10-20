"""배포 승인 판정 로직 단위 테스트 (07-4). MLflow 서버 없이 돈다."""

from src.registry import judge

GATE = {"min_test_f1": 0.85, "max_latency_ms": 20, "max_f1_drop": 0.0, "require_clean_git": True}
SLICE_GATE = GATE | {"min_slice_f1": {"night": 0.80}, "max_slice_drop": 0.02}


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


def test_missing_slice_is_not_a_pass():       # 12-5: 검사할 수 없는 기준은 통과로 치지 않는다
    checks = judge(tags(), None, SLICE_GATE)
    assert checks[-1] == ("night 조각이 테스트 세트에 없음", False)


def test_new_condition_model_passes():        # 12-5: 야간 조각을 배운 모델
    new = tags(f1="0.9100") | {"test_f1_day": "0.9200", "test_f1_night": "0.8700"}
    champ = {"test_f1_macro": "0.8600", "test_f1_day": "0.9259", "test_f1_night": "0.6000"}
    checks = judge(new, champ, SLICE_GATE)
    assert all(ok for _, ok in checks), checks


def test_forgetting_old_condition_is_rejected():   # 야간은 좋아졌지만 주간이 champion보다 0.02 넘게 떨어짐
    new = tags(f1="0.9000") | {"test_f1_day": "0.8900", "test_f1_night": "0.9000"}
    champ = {"test_f1_macro": "0.8600", "test_f1_day": "0.9259", "test_f1_night": "0.6000"}
    failed = [text for text, ok in judge(new, champ, SLICE_GATE) if not ok]
    assert len(failed) == 1 and failed[0].startswith("day 조각")
