"""재학습 후보 추천 에이전트 (13-4). 드리프트·품질 지표를 보고 재학습이 필요한지, 필요하면 어떤 데이터로 할지 추천한다.

    uv run python advise.py --window-start 2026-10-19T11:40:00Z --window-end 2026-10-19T12:01:00Z --repeat 3
추천과 근거는 MLflow 실험 retrain-recommendations에 실행 하나로 남는다. 실제로 재학습할지는 사람이 정한다.
"""

import argparse
import json
from pathlib import Path

import mlflow

from agent import grounding
from agent.core import run
from agent.tools_mlflow import model_versions, search_runs
from agent.tools_ops import alert_history, query, query_range
from agent.tools_retrain import RETRAIN_TOOLS, RETRAIN_TOOLS_PLUS

SYSTEM = """너는 꽃 분류 모델(flowers-classifier)의 재학습을 판단하는 에이전트다. 결정은 사람이 하고, 너는 추천과 근거를 쓴다.

알아 둘 것:
- 지표: flowers_input_brightness(입력 밝기 히스토그램), flowers_prediction_confidence(최고 확률 히스토그램),
  flowers_model_version. 드리프트 검사 결과는 MLflow 실험 flowers-drift(지표 drift_share, drifted_<특성>).
- 모델 버전 태그: test_f1_macro, test_f1_day, test_f1_night(어두운 사진 조각), champion_* (판정 때 같은 테스트 세트로 잰 champion).
- 학습 데이터의 어두운 사본은 make_night --fraction으로 만든다(원본 중 그 비율에 사본을 만든다). 분할은 data/split.csv로 고정.
- 판정 기준은 params.yaml의 gate.

규칙:
1. 재학습이 '필요한가'부터 판단한다: 모델 출력·품질이 나빠졌는가, 입력만 바뀌었는가.
2. 추천에는 바꿀 것(데이터 구성, 설정)을 구체적인 값으로 쓰고, 그 값을 고른 계산을 보인다.
3. 숫자는 도구 결과의 값만 쓴다. 계산한 값은 계산식을 쓴다.
4. 재학습 결과를 무엇으로 확인할지(게이트 기준, 배포 후 볼 지표)를 쓴다.

답 형식 (한국어): ## 판단 (재학습 필요: 예/아니오/보류) / ## 근거 / ## 추천 후보 / ## 확인 방법 / ## 위험"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-start", required=True)
    p.add_argument("--window-end", required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--label", default="default")
    p.add_argument("--with-production-summary", action="store_true", help="운영 구간 요약 도구를 더한다 (13-4 비교)")
    args = p.parse_args()
    task = (f"꽃 분류 서비스에서 v8 배포 뒤에도 입력 밝기 드리프트 알림(DataDrift, FlowersDarkInputs)이 계속되어 묵음을 걸어 두었다. "
            f"v8을 서비스한 구간은 {args.window_start} ~ {args.window_end} (UTC)이다. "
            "재학습이 필요한지 판단하고, 필요하면 어떤 데이터·설정으로 할지 후보를 추천해 줘.")
    tools = [query, query_range, alert_history, search_runs, model_versions] +         (RETRAIN_TOOLS_PLUS if args.with_production_summary else RETRAIN_TOOLS)
    out = Path("outputs/advise") / args.label
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.repeat + 1):
        mlflow.set_experiment("ch13-ops-agent")
        r = run(task, tools, SYSTEM, name="advise")
        g = grounding.check(r["answer"], r["messages"])
        mlflow.set_experiment("retrain-recommendations")          # 추천 근거 기록: 나중에 "왜 재학습했나"를 찾을 수 있게
        with mlflow.start_run(run_name=f"advise-{i}"):
            mlflow.set_tags({"agent_trace_id": r["trace_id"], "window": f"{args.window_start}~{args.window_end}",
                             "unsupported_numbers": len(g["unsupported"])})
            mlflow.log_text(r["answer"] or "", "recommendation.md")
            mlflow.log_metrics({"steps": r["steps"], "input_tokens": r["tokens"]["input"]})
        (out / f"{i}.md").write_text(r["answer"] or "(끝내지 못함)", encoding="utf-8")
        row = {"run": i, "steps": r["steps"], "tools": [c["tool"] for c in r["calls"]], "tokens": r["tokens"],
               "seconds": r["seconds"], "numbers": g["numbers"], "unsupported": g["unsupported"]}
        print(json.dumps(row, ensure_ascii=False))
        (out / "summary.jsonl").open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
