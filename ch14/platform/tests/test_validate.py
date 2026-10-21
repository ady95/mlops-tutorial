"""최종 점검의 드리프트 판정 (14-5). Prometheus 없이 돈다."""

import pytest

import validate

SPEC = {"run_job": "drift-flowers-run", "max_age_hours": 3}
H = 3600


def answers(ran=None, result=None, deferred=None, rows=None):
    return {'time() - push_time_seconds{job="drift-flowers-run"}': ran,
            'time() - push_time_seconds{job="drift-flowers"}': result,
            'ml_drift_check_deferred{job="drift-flowers-run"}': deferred,
            'ml_drift_check_rows{job="drift-flowers-run"}': rows}


@pytest.mark.parametrize("name, values, ok", [
    ("판단함", answers(ran=60, result=60, deferred=0, rows=338), True),
    ("판단 보류", answers(ran=60, result=8 * H, deferred=1, rows=0), True),
    ("검사 멈춤", answers(ran=4 * H, result=4 * H, deferred=0, rows=338), False),
    ("실행 기록 없음", answers(), False),
    ("판단했는데 결과가 오래됨", answers(ran=60, result=8 * H, deferred=0, rows=300), False),
    ("판단했는데 결과가 없음", answers(ran=60, result=None, deferred=0, rows=300), False),
])
def test_drift_status(monkeypatch, name, values, ok):
    monkeypatch.setattr(validate, "value", lambda q: values.get(q))
    passed, detail = validate.drift_status("drift-flowers", SPEC)
    assert passed is ok, detail


def test_deferred_shows_rows(monkeypatch):
    monkeypatch.setattr(validate, "value", lambda q: answers(ran=60, result=8 * H, deferred=1, rows=299).get(q))
    assert "표본 부족으로 판단 보류 (299건)" in validate.drift_status("drift-flowers", SPEC)[1]


def test_check_without_run_job_uses_result_time(monkeypatch):
    values = {'time() - push_time_seconds{job="drift-flower-support"}': 30 * H}
    monkeypatch.setattr(validate, "value", lambda q: values.get(q))
    assert validate.drift_status("drift-flower-support", {"max_age_hours": 24})[0] is False
