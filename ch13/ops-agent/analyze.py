"""실험 분석 에이전트 (13-3). MLflow 기록을 읽고 요약·비교·다음 실험 제안을 한다.

    uv run python analyze.py --question "ch04-flowers 실험에서 배포 후보를 골라 줘" --repeat 3
답에 나온 숫자가 도구 결과에 있었는지(grounding)를 함께 검사해 outputs/analyze/에 남긴다.
"""

import argparse
import json
from pathlib import Path

import mlflow

from agent import grounding
from agent.core import run
from agent.tools_mlflow import MLFLOW_TOOLS

SYSTEM = """너는 ML 실험 분석을 돕는 에이전트다. MLflow 기록만 근거로 쓴다.

규칙:
1. 숫자는 도구 결과에 나온 값만 쓴다. 계산한 값(차이, 평균)은 계산했다고 밝힌다.
2. 비교할 때는 조건이 같은지 먼저 본다: 데이터 해시(data_prepared_md5), 코드 상태(git_dirty), 장치(device·gpu 태그),
   평가에 쓴 세트(검증/테스트). 조건이 다르면 비교하지 않거나 그 차이를 적는다.
3. 한 번만 돌린 실행의 작은 차이는 우연일 수 있다. 같은 설정을 여러 번 돌린 기록이 있으면 그 폭을 함께 본다.
4. 결론의 확실도를 쓰고, 기록만으로 알 수 없는 것은 '기록에 없음'이라고 쓴다.

답 형식 (한국어): ## 요약 / ## 비교 (표) / ## 판단과 근거 / ## 주의할 점 / ## 다음 실험 제안 (2개, 각각 확인하려는 것)"""


SYSTEM_PLAIN = "너는 ML 실험 분석을 돕는 에이전트다. MLflow 기록을 보고 질문에 답한다. 한국어로 답한다."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--label", default="default")
    p.add_argument("--plain", action="store_true", help="비교 규칙 없이 (13-3: 규칙의 효과 비교)")
    args = p.parse_args()
    mlflow.set_experiment("ch13-ops-agent")
    out = Path("outputs/analyze") / args.label
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.repeat + 1):
        r = run(args.question, MLFLOW_TOOLS, SYSTEM_PLAIN if args.plain else SYSTEM, name="analyze")
        g = grounding.check(r["answer"], r["messages"])
        (out / f"{i}.md").write_text(r["answer"] or "(끝내지 못함)", encoding="utf-8")
        row = {"run": i, "steps": r["steps"], "tools": [c["tool"] for c in r["calls"]], "tokens": r["tokens"],
               "seconds": r["seconds"], "numbers": g["numbers"], "unsupported": g["unsupported"]}
        print(json.dumps(row, ensure_ascii=False))
        (out / "summary.jsonl").open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
