"""조사하고, 필요하면 조치를 '제안'하는 에이전트 (13-5). 실행은 사람이 승인 이슈에 /approve를 달아야 일어난다.

    uv run python respond.py --event "FlowersLowConfidence 발생 06:50Z" --start 2026-10-20T06:20:00Z --end 2026-10-20T07:00:00Z
    uv run python respond.py --event "flowers-classifier v9가 판정을 통과해 challenger가 됨"
환경변수: OPENAI_*, MLFLOW_TRACKING_URI, PROMETHEUS_URL, GITHUB_TOKEN(이슈 쓰기만), GITHUB_REPO
"""

import argparse
import json
from pathlib import Path

import mlflow

from agent.core import run
from agent.tools_action import ACTION_TOOLS
from agent.tools_mlflow import model_versions
from agent.tools_ops import OPS_TOOLS
from agent.tools_retrain import production_summary
from triage import SYSTEM as TRIAGE_SYSTEM

SYSTEM = TRIAGE_SYSTEM + """

조치 제안 규칙 (13-5):
- 조사 결과 사람이 할 조치가 분명하면 propose_action으로 승인 요청을 하나만 연다. 분명하지 않으면 열지 않고 조사할 것을 쓴다.
- 조치는 가장 작고 되돌리기 쉬운 것을 고른다. 모델 레지스트리(flowers-classifier)의 별칭·태그(approved_by, test_f1_*, champion_*)를 확인한다.
  - 새 버전을 champion으로: deploy (배포 판정을 다시 확인한다)
  - 승인을 거쳐 운영된 적 있는 버전(approved_by 태그)으로 되돌리기: restore
- 승인 요청의 근거에는 도구로 확인한 지표·시각·값을 쓰고, 틀렸을 때의 위험과 되돌리는 법을 쓴다.
- 너는 실행 권한이 없다. 승인 요청을 열었다면 그 번호를 답에 쓴다."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True)
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--label", default="default")
    args = p.parse_args()
    mlflow.set_experiment("ch13-ops-agent")
    task = f"일어난 일: {args.event}\n" + (f"조사할 구간(UTC): {args.start} ~ {args.end}\n" if args.start else "") + \
        "무슨 일인지 조사하고, 사람이 할 조치가 분명하면 승인 요청을 열어 줘."
    r = run(task, OPS_TOOLS + [model_versions, production_summary] + ACTION_TOOLS, SYSTEM, name="respond")
    out = Path("outputs/respond") / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer.md").write_text(r["answer"] or "(끝내지 못함)", encoding="utf-8")
    row = {"steps": r["steps"], "tools": [c["tool"] for c in r["calls"]], "tokens": r["tokens"], "seconds": r["seconds"],
           "proposals": [c["args"] for c in r["calls"] if c["tool"] == "propose_action"], "trace_id": r["trace_id"]}
    (out / "summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
