"""심판을 믿어도 되는지 점검한다: 일부러 망가뜨린 요약을 잡아내는가, 같은 답을 다시 내는가 (09-3).

실행:
    export OPENAI_API_KEY=... OPENAI_JUDGE_MODEL=...
    uv run python -m src.judge_check
"""

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.judge import complete, faithful

rows = [json.loads(line) for line in open("eval/evalset.jsonl", encoding="utf-8")]
rng = random.Random(0)
sample = rng.sample([r for r in rows if "적립금" not in r["inquiry"]], 40)

# 정답을 아는 시험 문제: (요약, 기대하는 faithful, 기대하는 complete)
cases = []
for r in sample:
    ref = r["reference_summary"]
    cases.append(("원본 요약", r, ref, True, True))
    cases.append(("없는 요청 추가", r, ref.rstrip(".") + ", 그리고 적립금 지급도 함께 요청했습니다.", False, None))
    cases.append(("핵심 요청 누락", r, "고객이 쇼핑몰에 문의를 남겼습니다.", None, False))
    oid = r["gold"]["order_id"]
    if oid and oid in ref:     # 눈에 덜 띄는 오류: 주문번호 한 자리만 바꾼다
        wrong = oid[:-1] + str((int(oid[-1]) + 1) % 10)
        cases.append(("주문번호 한 자리", r, ref.replace(oid, wrong), False, None))


def run(case):
    kind, r, summary, _, _ = case
    inputs, outputs = {"inquiry": r["inquiry"]}, {"summary": summary}
    return faithful(inputs=inputs, outputs=outputs).value, complete(inputs=inputs, outputs=outputs).value


with ThreadPoolExecutor(8) as pool:
    first = list(pool.map(run, cases))
    second = list(pool.map(run, cases))      # 같은 입력으로 한 번 더


def as_bool(v):
    return str(v).lower() == "true"


for kind in ("원본 요약", "없는 요청 추가", "핵심 요청 누락", "주문번호 한 자리"):
    idx = [i for i, c in enumerate(cases) if c[0] == kind]
    exp_f, exp_c = cases[idx[0]][3], cases[idx[0]][4]
    line = f"{kind:10s} {len(idx)}건:"
    if exp_f is not None:
        line += f" faithful 기대대로 {sum(as_bool(first[i][0]) == exp_f for i in idx)}/{len(idx)}"
    if exp_c is not None:
        line += f" | complete 기대대로 {sum(as_bool(first[i][1]) == exp_c for i in idx)}/{len(idx)}"
    print(line)
for j, crit in enumerate(("faithful", "complete")):
    same = sum(as_bool(first[i][j]) == as_bool(second[i][j]) for i in range(len(cases)))
    flips = [cases[i][0] for i in range(len(cases)) if as_bool(first[i][j]) != as_bool(second[i][j])]
    print(f"두 번 채점해 같은 판정 ({crit}): {same}/{len(cases)}  달라진 건: {dict(Counter(flips))}")
