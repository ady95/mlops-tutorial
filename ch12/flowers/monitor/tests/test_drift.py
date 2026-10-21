"""드리프트 검사의 실행 기록 순서 (12-3, 14-5). 서버 없이 돈다: Evidently·MLflow·Pushgateway를 흉내 낸다.

    cd monitor && uv run --with pytest pytest -q
"""

import contextlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drift  # noqa: E402


class FakeSnapshot:
    def dict(self):
        return {"metrics": [{"value": {"share": 0.2}},
                            {"config": {"column": "brightness", "method": "wasserstein", "threshold": 0.1},
                             "value": 0.05}]}

    def save_html(self, path):
        Path(path).write_text("<html></html>", encoding="utf-8")


def gauge_value(registry, name):
    for metric in registry.collect():
        if metric.name == name:
            return metric.samples[0].value
    return None


@pytest.fixture
def pushes(monkeypatch, tmp_path):
    """push_to_gateway 호출을 (job, 레지스트리)로 기록한다. fail_job에 든 job은 HTTP 500처럼 실패한다."""
    log, fail_job = [], set()

    def fake_push(gateway, job, registry):
        if job in fail_job:
            raise OSError("HTTP 500")
        log.append((job, registry))

    monkeypatch.setattr(drift, "push_to_gateway", fake_push)
    monkeypatch.setattr(drift, "check", lambda current, reference: FakeSnapshot())
    monkeypatch.setattr(drift, "load_reference", lambda: ("11", pd.DataFrame()))
    for name in ("set_experiment", "set_tags", "log_metrics", "log_artifact"):
        monkeypatch.setattr(drift.mlflow, name, lambda *a, **k: None)
    monkeypatch.setattr(drift.mlflow, "start_run", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
    return log, fail_job


def rows(monkeypatch, n):
    monkeypatch.setattr(drift, "load_current", lambda log_dir, hours: pd.DataFrame({"model_version": ["11"] * n}))


def test_too_few_rows_records_deferred_run_only(monkeypatch, pushes):
    log, _ = pushes
    rows(monkeypatch, drift.config["min_rows"] - 1)
    drift.main()
    assert [job for job, _ in log] == ["drift-flowers-run"]          # 결과는 건드리지 않는다
    assert gauge_value(log[0][1], "ml_drift_check_deferred") == 1


def test_judged_run_records_result_then_run(monkeypatch, pushes):
    log, _ = pushes
    rows(monkeypatch, drift.config["min_rows"])
    drift.main()
    assert [job for job, _ in log] == ["drift-flowers", "drift-flowers-run"]
    assert gauge_value(log[1][1], "ml_drift_check_deferred") == 0


def test_failed_result_push_leaves_run_record_untouched(monkeypatch, pushes):
    log, fail_job = pushes
    fail_job.add("drift-flowers")
    rows(monkeypatch, drift.config["min_rows"])
    with pytest.raises(OSError):
        drift.main()
    assert log == []                     # 실행 기록이 갱신되지 않아야 DriftCheckStale이 3시간 뒤 잡는다
