"""LLM 서비스의 드리프트 검사 (12-3). 트레이스(11장)에서 입력·출력 특성을 뽑아 두 구간을 비교한다.

프로젝트 A는 '학습 데이터'를 기준으로 삼았다. LLM 서비스는 학습 데이터(합성 문의)와 실제 문의의 말투가 처음부터
다르므로, 지금 모델·프롬프트로 안정적으로 운영된 '지난 구간'을 기준으로 삼는다.

학습 환경(torch 등)과 분리된 가벼운 환경에서, 모니터링 서버(Pushgateway 옆)에서 돈다.
실행 (monitor 폴더에서):
    uv run python drift_llm.py --reference-hours 24 --current-hours 6
    uv run python drift_llm.py --phases load_phases.json        # load.py가 남긴 구간끼리 (첫 구간 = 기준)
"""

import argparse
import json
import tempfile
import time
from datetime import datetime

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

NUMERIC = ["inquiry_chars", "category_confidence"]
CATEGORICAL = ["category", "channel", "product_status", "has_order_id"]

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="ch12-flower-support-serving")
parser.add_argument("--phases", help="load.py 구간 파일: 첫 구간이 기준, 마지막 구간이 현재")
parser.add_argument("--reference-hours", type=float, default=24)
parser.add_argument("--current-hours", type=float, default=6)
parser.add_argument("--pushgateway", default="localhost:9091")
args = parser.parse_args()


def features(start_ms: int, end_ms: int) -> pd.DataFrame:
    exp_id = mlflow.get_experiment_by_name(args.experiment).experiment_id
    traces = mlflow.search_traces(locations=[exp_id], max_results=20000, return_type="list",
                                  filter_string=f"trace.timestamp_ms >= {start_ms} AND trace.timestamp_ms <= {end_ms}")
    rows = []
    for t in traces:
        root = next(s for s in t.data.spans if s.parent_id is None)
        result = (root.outputs or {}).get("result") or {}
        attrs = root.attributes
        product = result.get("product")
        rows.append({
            "inquiry_chars": len((root.inputs or {}).get("inquiry", "")),
            "category_confidence": attrs.get("category_confidence"),
            "category": result.get("category", "invalid"),
            "channel": t.info.tags.get("channel", "?"),
            # 상품: 언급 없음 / 카탈로그에 있음 / 카탈로그에 없음 (11-4의 신호를 범주로)
            "product_status": "none" if product is None else
                              "in_catalog" if attrs.get("product_in_catalog", True) else "not_in_catalog",
            "has_order_id": str(result.get("order_id") is not None)})
    return pd.DataFrame(rows)


def windows() -> tuple[tuple[int, int], tuple[int, int]]:
    if args.phases:
        phases = json.load(open(args.phases, encoding="utf-8"))
        return (phases[0]["start_ms"], phases[0]["end_ms"]), (phases[-1]["start_ms"], phases[-1]["end_ms"])
    now = int(time.time() * 1000)
    cur = int(args.current_hours * 3_600_000)
    return (now - cur - int(args.reference_hours * 3_600_000), now - cur), (now - cur, now)


def main():
    (ref_start, ref_end), (cur_start, cur_end) = windows()
    reference, current = features(ref_start, ref_end), features(cur_start, cur_end)
    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    # 표본이 수천 건이면 p-value 검정은 아주 작은 차이도 '다름'으로 본다 — 차이의 크기(거리)로 판정한다 (12-3)
    preset = DataDriftPreset(num_method="wasserstein", cat_method="jensenshannon", threshold=0.1)
    snapshot = Report([preset]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition))
    metrics = snapshot.dict()["metrics"]
    share = metrics[0]["value"]["share"]
    drifted = {}
    for m in metrics[1:]:
        cfg = m["config"]
        score = float(m["value"])
        drifted[cfg["column"]] = (score < cfg["threshold"]) if "p_value" in cfg["method"] else (score > cfg["threshold"])
        print(f"  {cfg['column']:20} {cfg['method']:28} {score:.4g}  {'다름' if drifted[cfg['column']] else '같음'}")

    mlflow.set_experiment("flower-support-drift")
    with mlflow.start_run(run_name=datetime.now().strftime("drift-%Y%m%d-%H%M")):
        mlflow.log_metrics({"drift_share": share, "reference_rows": len(reference), "current_rows": len(current)} |
                           {f"drifted_{c}": int(v) for c, v in drifted.items()})
        with tempfile.TemporaryDirectory() as tmp:
            snapshot.save_html(f"{tmp}/drift_report.html")
            mlflow.log_artifact(f"{tmp}/drift_report.html")
            for name, df in (("reference", reference), ("current", current)):   # 범주별 비율을 사람이 바로 보도록
                df.to_csv(f"{tmp}/{name}.csv", index=False)
                mlflow.log_artifact(f"{tmp}/{name}.csv")

    registry = CollectorRegistry()
    Gauge("ml_drift_share", "드리프트로 판정된 특성의 비율", ["project"], registry=registry) \
        .labels(project="flower-support").set(share)
    g = Gauge("ml_feature_drifted", "특성별 드리프트 판정 (1=다름)", ["project", "feature"], registry=registry)
    for c, v in drifted.items():
        g.labels(project="flower-support", feature=c).set(int(v))
    push_to_gateway(args.pushgateway, job="drift-flower-support", registry=registry)
    print(f"기준 {len(reference)}건 vs 현재 {len(current)}건: 드리프트 특성 비율 {share:.2f}")
    print("상품 상태 비율:")
    print(pd.concat({"기준": reference["product_status"].value_counts(normalize=True),
                     "현재": current["product_status"].value_counts(normalize=True)}, axis=1).round(3).to_string())


if __name__ == "__main__":
    main()
