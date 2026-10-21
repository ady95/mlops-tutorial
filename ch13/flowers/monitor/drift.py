"""꽃 분류 서비스의 드리프트 검사 (12-3). 매시간 monitor 워크플로가 실행한다.

1. 추론 서버의 예측 기록(최근 window_hours시간)을 읽는다            → 현재 데이터
2. 지금 champion에 붙은 기준 데이터(reference_run_id)를 받는다      → 기준 데이터
3. Evidently로 특성마다 분포가 같은지 검정한다
4. 결과 숫자는 Pushgateway로 (Prometheus 알림), 리포트(HTML)는 MLflow로 (사람이 원인 확인)
5. 실행 기록은 결과와 다른 job(drift-flowers-run)으로 — 표본이 모자라 판단을 미룬 실행도 남긴다 (14-5).
   판단하는 실행은 결과를 MLflow·Pushgateway에 모두 남긴 뒤에야 기록한다. 중간에 실패하면 기록이 갱신되지 않아
   DriftCheckStale이 잡는다 (14장 검수에서 고침)

실행 (monitor 폴더에서):
    PREDICTION_LOG_DIR=~/work12/flowers-api/logs uv run python drift.py
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

NAME = "flowers-classifier"
config = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))
NUMERIC, CATEGORICAL = config["numeric"], config["categorical"]


def load_current(log_dir: str, hours: int) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    days = {(since + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((hours // 24) + 2)}
    rows = []
    for day in sorted(days):
        path = Path(log_dir) / f"predictions-{day}.jsonl"
        if path.exists():
            rows += [json.loads(line) for line in path.open(encoding="utf-8")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[pd.to_datetime(df["ts"]) >= since]


def load_reference() -> tuple[str, pd.DataFrame]:
    mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, "champion")
    run_id = mv.tags.get("reference_run_id")
    if not run_id:
        sys.exit(f"champion v{mv.version}에 기준 데이터가 없습니다: uv run python -m src.reference {mv.version}")
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference.csv")
    return mv.version, pd.read_csv(path)


def push_run(gateway: str, rows: int, deferred: bool):
    """검사가 돌았다는 기록. 결과만 남기면, 요청이 적어 판단을 미룬 것과 검사가 멈춘 것을 알림이 구분하지 못한다 (14-5)."""
    registry = CollectorRegistry()
    labels = {"project": "flowers"}
    Gauge("ml_drift_check_rows", "이번 검사가 본 예측 수", ["project"], registry=registry).labels(**labels).set(rows)
    Gauge("ml_drift_check_deferred", "표본이 모자라 판단을 미뤘는가 (1=미룸)", ["project"],
          registry=registry).labels(**labels).set(int(deferred))
    push_to_gateway(gateway, job="drift-flowers-run", registry=registry)


def check(current: pd.DataFrame, reference: pd.DataFrame):
    """특성마다 두 분포를 비교하고, '다름'인 특성이 drift_share 이상이면 데이터셋 드리프트."""
    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    preset = DataDriftPreset(drift_share=config["drift_share"], num_method=config["num_method"],
                             cat_method=config["cat_method"], threshold=config["threshold"])
    return Report([preset]).run(Dataset.from_pandas(current[NUMERIC + CATEGORICAL], data_definition=definition),
                                Dataset.from_pandas(reference[NUMERIC + CATEGORICAL], data_definition=definition))


def main():
    current = load_current(os.path.expanduser(os.environ["PREDICTION_LOG_DIR"]), config["window_hours"])
    version, reference = load_reference()
    if not current.empty:   # 예측·확신도는 모델마다 다르다 — 지금 champion이 낸 예측만 비교한다 (12-5)
        current = current[current["model_version"].astype(str) == str(version)]
    if len(current) < config["min_rows"]:          # 표본이 적으면 검정 결과를 믿을 수 없다 — 판단을 미룬다
        push_run(config["pushgateway"], len(current), deferred=True)
        print(f"최근 {config['window_hours']}시간 예측 {len(current)}건 < {config['min_rows']}건: 판단을 미룹니다 (실행 기록만 남김)")
        return
    snapshot = check(current, reference)

    # 결과 읽기: 첫 지표는 드리프트 특성 수·비율, 나머지는 특성별 검정 (값과 기준)
    metrics = snapshot.dict()["metrics"]
    share = metrics[0]["value"]["share"]
    features = {}
    for m in metrics[1:]:
        cfg = m["config"]
        score = float(m["value"])
        drifted = score < cfg["threshold"] if "p_value" in cfg["method"] else score > cfg["threshold"]
        features[cfg["column"]] = {"score": score, "method": cfg["method"], "drifted": drifted}

    # 1) 사람이 볼 리포트 → MLflow
    mlflow.set_experiment("flowers-drift")
    with mlflow.start_run(run_name=datetime.now().strftime("drift-%Y%m%d-%H%M")):
        mlflow.set_tags({"champion_version": version, "window_hours": config["window_hours"]})
        mlflow.log_metrics({"drift_share": share, "rows": len(current)} |
                           {f"drifted_{c}": int(f["drifted"]) for c, f in features.items()})
        with tempfile.TemporaryDirectory() as tmp:
            snapshot.save_html(f"{tmp}/drift_report.html")
            mlflow.log_artifact(f"{tmp}/drift_report.html")

    # 2) 알림용 숫자 → Pushgateway (job 이름이 같으면 이전 값을 덮어쓴다)
    registry = CollectorRegistry()
    labels = {"project": "flowers"}
    Gauge("ml_drift_share", "드리프트로 판정된 특성의 비율", ["project"], registry=registry).labels(**labels).set(share)
    Gauge("ml_drift_rows", "검사한 예측 수", ["project"], registry=registry).labels(**labels).set(len(current))
    g = Gauge("ml_feature_drifted", "특성별 드리프트 판정 (1=다름)", ["project", "feature"], registry=registry)
    for c, f in features.items():
        g.labels(feature=c, **labels).set(int(f["drifted"]))
    push_to_gateway(config["pushgateway"], job="drift-flowers", registry=registry)
    # 3) 결과를 모두 남긴 뒤에야 실행 기록 — 위에서 실패하면 기록이 갱신되지 않아 DriftCheckStale이 잡는다
    push_run(config["pushgateway"], len(current), deferred=False)

    print(f"champion v{version} 기준 {len(reference)}행 vs 최근 {len(current)}건: 드리프트 특성 비율 {share:.2f}")
    for c, f in features.items():
        print(f"  {c:16} {f['method']:22} {f['score']:.4g}  {'다름' if f['drifted'] else '같음'}")


if __name__ == "__main__":
    main()
