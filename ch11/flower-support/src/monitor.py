"""운영 기록(트레이스)으로 서비스 상태를 점검한다: 사용량 집계, 규칙 신호, 표본 요약 심판, 알림 (11-1, 11-3, 11-4).

실행 (API 키가 있는 곳에서 — 요약 심판을 부른다):
    export MLFLOW_TRACKING_URI=... OPENAI_API_KEY=... OPENAI_JUDGE_MODEL=...
    uv run python -m src.monitor --phases eval/outputs/simulate_phases.json --sample 40
    uv run python -m src.monitor --minutes 60            # 최근 60분
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import mlflow
import yaml
from mlflow.entities import AssessmentSource

from src.judge import JUDGE_MODEL, complete, faithful

parser = argparse.ArgumentParser()
parser.add_argument("--phases", help="simulate.py가 남긴 구간 파일")
parser.add_argument("--minutes", type=int, help="최근 N분을 한 구간으로")
parser.add_argument("--sample", type=int, default=40, help="구간마다 요약 심판에 넘길 표본 수")
parser.add_argument("--experiment", default="ch11-flower-support-serving")
args = parser.parse_args()

rules = yaml.safe_load(open("params.yaml", encoding="utf-8"))["monitor"]
exp_id = mlflow.get_experiment_by_name(args.experiment).experiment_id


def load(start_ms: int, end_ms: int) -> list[dict]:
    traces = mlflow.search_traces(locations=[exp_id], max_results=5000, return_type="list",
                                  filter_string=f"trace.timestamp_ms >= {start_ms} AND trace.timestamp_ms <= {end_ms}")
    rows = []
    for t in traces:
        root = next(s for s in t.data.spans if s.parent_id is None)
        usage = t.info.token_usage or {}
        rows.append({"trace_id": t.info.trace_id, "inquiry": (root.inputs or {}).get("inquiry", ""),
                     "result": (root.outputs or {}).get("result"), **t.info.tags,
                     "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                     "latency_ms": t.info.execution_duration, **root.attributes})
    return rows


def judge_sample(rows: list[dict], n: int) -> float:
    """표본의 요약을 9장 심판으로 채점하고, 판정을 그 트레이스에 피드백으로 남긴다."""
    sample = random.Random(0).sample([r for r in rows if r["result"]], min(n, len(rows)))

    def one(r):
        inputs, outputs = {"inquiry": r["inquiry"]}, {"summary": r["result"].get("summary", "")}
        ok = True
        for judge in (faithful, complete):
            fb = judge(inputs=inputs, outputs=outputs)
            mlflow.log_feedback(trace_id=r["trace_id"], name=judge.name, value=fb.value, rationale=fb.rationale,
                                source=AssessmentSource(source_type="LLM_JUDGE", source_id=JUDGE_MODEL))
            ok = ok and str(fb.value).lower() == "true"
        return ok

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, sample))
    return sum(results) / len(results)


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    rate = lambda key: sum(bool(r.get(key)) for r in rows) / n  # noqa: E731
    low = sum((r.get("category_confidence") or 0) < rules["low_confidence"] for r in rows) / n
    m = {"requests": n, "input_tokens": sum(r["input_tokens"] for r in rows),
         "output_tokens": sum(r["output_tokens"] for r in rows),
         "schema_valid": rate("schema_valid"), "order_id_grounded": rate("order_id_grounded"),
         "product_in_inquiry": rate("product_in_inquiry"), "product_in_catalog": rate("product_in_catalog"),
         "low_confidence_rate": low,
         "product_null_rate": sum(bool(r["result"]) and r["result"].get("product") is None for r in rows) / n,
         "injection_suspected": rate("injection_suspected"),
         "pii_masked": sum(bool(r.get("pii_masked")) for r in rows),
         "latency_p50_ms": sorted(r["latency_ms"] for r in rows)[n // 2]}
    m["summary_pass"] = judge_sample(rows, args.sample)
    by_channel = defaultdict(lambda: [0, 0])
    for r in rows:
        by_channel[r.get("channel", "?")][0] += 1
        by_channel[r.get("channel", "?")][1] += r["input_tokens"] + r["output_tokens"]
    versions = Counter(f"모델 v{r.get('model_version')} + 프롬프트 v{r.get('prompt_version')}" for r in rows)
    print(f"\n[{name}] 요청 {n}건 ({', '.join(f'{k} {v}' for k, v in versions.items())})")
    print(f"  토큰: 입력 {m['input_tokens']:,} / 출력 {m['output_tokens']:,} "
          f"(요청당 {m['input_tokens'] / n:.0f} / {m['output_tokens'] / n:.0f})")
    print("  경로별: " + ", ".join(f"{c} {k}건 {t:,}토큰" for c, (k, t) in sorted(by_channel.items())))
    for key in ("schema_valid", "order_id_grounded", "product_in_inquiry", "product_in_catalog", "product_null_rate",
                "low_confidence_rate",
                "injection_suspected", "summary_pass"):
        print(f"  {key:22s} {m[key]:.3f}")
    print(f"  민감정보 가림 {m['pii_masked']}건, 지연 p50 {m['latency_p50_ms']}ms")
    return m


def alerts(m: dict) -> list[str]:
    out = []
    if m["schema_valid"] < rules["min_schema_valid"]:
        out.append(f"형식 오류 증가: {m['schema_valid']:.3f} < {rules['min_schema_valid']}")
    for key in ("order_id_grounded", "product_in_inquiry"):
        if m[key] < rules["min_grounded"]:
            out.append(f"환각 의심 증가({key}): {m[key]:.3f} < {rules['min_grounded']}")
    if m["product_in_catalog"] < rules["min_grounded"]:
        out.append(f"카탈로그에 없는 상품 증가: {m['product_in_catalog']:.3f} < {rules['min_grounded']} (새 상품인지 확인)")
    if m["low_confidence_rate"] > rules["max_low_confidence_rate"]:
        out.append(f"낮은 확신도 증가: {m['low_confidence_rate']:.3f} > {rules['max_low_confidence_rate']}")
    if m["summary_pass"] < rules["min_summary_pass"]:
        out.append(f"요약 품질 저하: {m['summary_pass']:.3f} < {rules['min_summary_pass']}")
    return out


if args.phases:
    windows = [(p["phase"], p["start_ms"], p["end_ms"]) for p in json.load(open(args.phases, encoding="utf-8"))]
else:
    now = int(time.time() * 1000)
    windows = [(f"최근 {args.minutes}분", now - args.minutes * 60_000, now)]

mlflow.set_experiment("ch11-monitoring")
for name, start, end in windows:
    m = summarize(name, load(start, end))
    found = alerts(m)
    print("  알림: " + ("; ".join(found) if found else "없음"))
    with mlflow.start_run(run_name=f"monitor-{name}"):
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        mlflow.set_tags({"window_start_ms": start, "window_end_ms": end, "alerts": len(found)})
        mlflow.log_dict({"alerts": found}, "alerts.json")
