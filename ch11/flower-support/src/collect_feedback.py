"""상담원이 '틀렸다'고 표시하고 바로잡은 운영 요청을 모아, 평가 세트 형식으로 내보낸다 (11-5).

내보낸 파일은 사람이 검토한 뒤 eval/ 아래 새 조각으로 더하고(9장 방식), 일부는 학습 데이터 보강에 쓴다.

실행:
    uv run python -m src.collect_feedback --days 7
"""

import argparse
import json
import time
from pathlib import Path

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=7)
parser.add_argument("--experiment", default="ch11-flower-support-serving")
parser.add_argument("--out", default="eval/feedback_cases.jsonl")
args = parser.parse_args()

exp_id = mlflow.get_experiment_by_name(args.experiment).experiment_id
since = int((time.time() - args.days * 86400) * 1000)
traces = mlflow.search_traces(locations=[exp_id], max_results=10000, return_type="list",
                              filter_string=f"trace.timestamp_ms >= {since}")
cases = []
for t in traces:
    assessments = {a.name: a for a in t.info.assessments}
    if "user_correct" not in assessments or assessments["user_correct"].value:
        continue
    root = next(s for s in t.data.spans if s.parent_id is None)
    expected = assessments.get("expected_result")
    cases.append({"id": t.info.trace_id, "slice": "운영 피드백", "inquiry": root.inputs["inquiry"],
                  "model_output": (root.outputs or {}).get("result"),
                  "gold": expected.value if expected else None,
                  "model_version": t.info.tags.get("model_version"), "prompt_version": t.info.tags.get("prompt_version"),
                  "comment": assessments["user_correct"].rationale})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8", newline="\n") as f:
    for c in cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"최근 {args.days}일 트레이스 {len(traces)}건 중 '틀림' 표시 {len(cases)}건 "
      f"(바로잡은 값 있음 {sum(c['gold'] is not None for c in cases)}건) → {args.out}")
