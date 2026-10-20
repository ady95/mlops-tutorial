"""재학습 판단 도구 — 읽기 전용 (13-4). 모델 버전의 기준 데이터 요약, 파이프라인 설정·보고서.

설정: PIPELINE_DIR (비공개 파이프라인 저장소를 받아 둔 폴더)
"""

import csv
import io
import os
from pathlib import Path

import mlflow

from agent.core import tool

PIPELINE = Path(os.environ.get("PIPELINE_DIR", "../flowers")).resolve()
READABLE = {"params.yaml", "dvc.yaml", "reports/prepare.json", "reports/validation.json", "monitor/config.yaml"}


@tool("모델 버전에 붙은 드리프트 기준 데이터(검증 세트의 특성·예측) 요약: 행 수, 밝기 0.2 미만 비율, 평균 확신도 등.",
      version={"type": "string"})
def reference_summary(version: str) -> dict:
    mv = mlflow.MlflowClient().get_model_version("flowers-classifier", version)
    run_id = mv.tags.get("reference_run_id")
    if not run_id:
        return f"v{version}에는 기준 데이터가 없음"
    text = mlflow.artifacts.load_text(f"runs:/{run_id}/reference.csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    n = len(rows)
    dark = sum(float(r["brightness"]) < 0.2 for r in rows)
    copies = sum(r["path"].split("/")[-1].startswith("night_") for r in rows)
    return {"version": version, "rows": n, "dark_ratio": round(dark / n, 4), "night_copies": copies,
            "mean_confidence": round(sum(float(r["confidence"]) for r in rows) / n, 4),
            "low_confidence_ratio": round(sum(float(r["confidence"]) < 0.6 for r in rows) / n, 4)}


@tool("파이프라인 저장소의 설정·보고서 파일을 읽는다. 읽을 수 있는 파일: " + ", ".join(sorted(READABLE)),
      path={"type": "string"})
def pipeline_file(path: str) -> str:
    if path not in READABLE:
        return f"읽을 수 없는 파일: {path}"
    return (PIPELINE / path).read_text(encoding="utf-8")


@tool("운영 구간의 입력·출력 요약: 요청 수, 밝기 0.2 미만 사진 비율, 확신도 0.6 미만 비율, 평균 밝기. "
      "reference_summary와 같은 기준으로 계산하므로 둘을 바로 비교할 수 있다.",
      start={"type": "string", "description": "ISO 시각"}, end={"type": "string", "description": "ISO 시각"})
def production_summary(start: str, end: str) -> dict:
    from agent.tools_ops import query, ts
    window = f"{int(ts(end) - ts(start))}s"

    def one(promql: str):
        return (query.fn(promql, at=end) or [{"value": None}])[0]["value"]

    n = one(f"sum(increase(flowers_input_brightness_count[{window}]))")
    dark = one(f'sum(increase(flowers_input_brightness_bucket{{le="0.2"}}[{window}]))')
    low = one(f'sum(increase(flowers_prediction_confidence_bucket{{le="0.6"}}[{window}]))')
    bsum = one(f"sum(increase(flowers_input_brightness_sum[{window}]))")
    if not n:
        return "이 구간에 요청이 없음"
    return {"requests": round(n), "dark_ratio": round(dark / n, 4), "low_confidence_ratio": round(low / n, 4),
            "mean_brightness": round(bsum / n, 4), "note": "히스토그램 구간 기준(밝기 0.2 이하, 확신도 0.6 이하)"}


RETRAIN_TOOLS = [reference_summary, pipeline_file]
RETRAIN_TOOLS_PLUS = RETRAIN_TOOLS + [production_summary]     # 13-4: 운영 쪽 같은 기준의 요약을 더한 판
