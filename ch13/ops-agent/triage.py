"""장애 분석 에이전트 (13-2). 알림 하나와 시간 구간을 주면, 지표·알림 이력·로그를 읽고 원인을 추정한다.

    uv run python triage.py --alert "ServiceDown llm-gateway 11:00Z" --start 2026-10-19T10:50:00Z --end 2026-10-19T11:10:00Z
    uv run python triage.py --cases cases/incidents.yaml --repeat 3          # 12장의 실제 장애 다시 보기
결과는 outputs/triage/<사례>-<회차>.md 에 남고, 실행 전체는 MLflow 트레이스(실험 ch13-ops-agent)에 남는다.
"""

import argparse
import json
from pathlib import Path

import mlflow
import yaml

from agent.core import run
from agent.tools_ops import OPS_TOOLS

SYSTEM = """너는 MLOps 운영 보조 에이전트다. 알림을 받은 당직자를 돕는다.

서비스 구성:
- flowers-api: 꽃 사진 분류 추론 서버(모니터링 서버와 같은 곳). 지표 flowers_*.
  flowers_input_brightness(입력 밝기 히스토그램), flowers_prediction_confidence(최고 확률 히스토그램), flowers_model_version(게이지)
- llm-gateway: 고객 문의 LLM 게이트웨이(server="llm"인 GPU 서버). 지표 llm_*. vLLM 지표는 게이트웨이의 /vllm-metrics로 중계(job="vllm")
- dcgm, node-gpu: GPU 서버의 GPU·서버 지표. node: 모니터링 서버 자신
- pushgateway: 드리프트 검사(job=drift-flowers, drift-flower-support)가 결과를 맡기는 곳. push_time_seconds = 마지막으로 맡긴 시각
- up{job=...} = 1이면 그 대상을 읽는 데 성공, scrape_duration_seconds = 읽는 데 걸린 시간(시간 초과 10초)

규칙:
1. 추측하기 전에 도구로 확인한다. 알림 구간 앞뒤를 함께 본다.
2. 근거는 도구 결과에 실제로 나온 숫자와 시각만 쓴다. 도구로 확인하지 않은 것은 '확인하지 못함'이라고 쓴다.
3. 원인이 여럿 가능하면 가능성 순으로 쓰고, 각각 무엇을 보면 가릴 수 있는지 쓴다.
4. 너는 아무것도 바꾸지 않는다. 사람이 할 일을 제안만 한다.

답 형식 (한국어):
## 요약 (두 문장)
## 근거 (지표·시각·값)
## 원인 판단 (확실도: 높음/중간/낮음)
## 확인하지 못한 것
## 사람이 할 일"""


def triage(alert: str, start: str, end: str, name: str) -> dict:
    task = f"알림: {alert}\n조사할 구간(UTC): {start} ~ {end}\n무슨 일이 있었고 원인이 무엇인지 조사해 줘."
    return run(task, OPS_TOOLS, SYSTEM, name=name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alert")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--cases")
    p.add_argument("--only", help="사례 id (쉼표로 여럿)")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--label", default="default", help="결과 폴더 이름 (모델을 바꿔 비교할 때)")
    args = p.parse_args()
    mlflow.set_experiment("ch13-ops-agent")
    out = Path("outputs/triage") / args.label
    out.mkdir(parents=True, exist_ok=True)
    cases = yaml.safe_load(open(args.cases, encoding="utf-8")) if args.cases else \
        [{"id": "adhoc", "alert": args.alert, "start": args.start, "end": args.end}]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    summary = []
    for c in cases:
        for i in range(1, args.repeat + 1):
            r = triage(c["alert"], c["start"], c["end"], name=f"triage-{c['id']}")
            (out / f"{c['id']}-{i}.md").write_text(r["answer"] or "(끝내지 못함)", encoding="utf-8")
            row = {"case": c["id"], "run": i, "steps": r["steps"], "tool_calls": len(r["calls"]),
                   "tools": [x["tool"] for x in r["calls"]], "tokens": r["tokens"], "seconds": r["seconds"],
                   "trace_id": r["trace_id"]}
            summary.append(row)
            print(json.dumps(row, ensure_ascii=False))
    (out / "summary.jsonl").open("a", encoding="utf-8").write("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in summary))


if __name__ == "__main__":
    main()
