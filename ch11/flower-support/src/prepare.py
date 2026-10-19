"""합성 데이터 품질 검사 → 층화 분할 → TRL 학습 형식 변환 (08-1).

입력: data/raw/generated.jsonl (make_dataset.py)
출력: data/prepared/{train,val,test}.jsonl, reports/prepare.json
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from schema import FIELDS, ORDER_ID, SYSTEM_PROMPT  # noqa: E402

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["prepare"]
PHONE = re.compile(r"01[016789]-?\d{3,4}-?\d{4}")


def problems(r: dict) -> list[str]:
    """한 건의 품질 문제 목록. 비어 있으면 통과."""
    found = []
    if any(r["check"][k] != r[k] for k in ("category", "urgency", "sentiment")):
        found.append("검수 불일치")
    if set(ORDER_ID.findall(r["inquiry"])) != ({r["order_id"]} if r["order_id"] else set()):
        found.append("주문번호 불일치")
    if r["product"] and r["product"] not in r["inquiry"]:
        found.append("상품명 누락")
    if PHONE.search(r["inquiry"]):
        found.append("전화번호 포함")
    if not 10 <= len(r["inquiry"]) <= 600:
        found.append("길이 이상")
    return found


def to_example(r: dict) -> dict:
    target = {k: r[k] for k in FIELDS}
    return {
        "id": r["id"],
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["inquiry"]}],
        "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
        "chat_template_kwargs": {"enable_thinking": False},   # 생각 모드를 끈 형식으로 학습·추론 (08-3)
    }


rows = [json.loads(line) for line in open("data/raw/generated.jsonl", encoding="utf-8")]
rows.sort(key=lambda r: r["id"])
report, kept, seen = Counter(), [], set()
for r in rows:
    found = problems(r)
    if r["inquiry"] in seen:
        found.append("중복")
    seen.add(r["inquiry"])
    report.update(found)
    if not found:
        kept.append(r)

# 문의 유형별로 같은 비율이 되게 나눈다 (3장 prepare.py와 같은 방식)
rng = random.Random(params["seed"])
by_cat = defaultdict(list)
for r in kept:
    by_cat[r["category"]].append(r)
splits = {"train": [], "val": [], "test": []}
for cat, items in sorted(by_cat.items()):
    rng.shuffle(items)
    n_test = round(len(items) * params["test_size"] / len(kept))
    n_val = round(len(items) * params["val_size"] / len(kept))
    splits["test"] += items[:n_test]
    splits["val"] += items[n_test:n_test + n_val]
    splits["train"] += items[n_test + n_val:]

Path("data/prepared").mkdir(parents=True, exist_ok=True)
for name, items in splits.items():
    with open(f"data/prepared/{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:   # 윈도에서도 LF
        for r in sorted(items, key=lambda r: r["id"]):
            f.write(json.dumps(to_example(r), ensure_ascii=False) + "\n")

summary = {"generated": len(rows), "kept": len(kept), "dropped": dict(report),
           **{f"n_{k}": len(v) for k, v in splits.items()},
           "category_counts": dict(Counter(r["category"] for r in kept))}
Path("reports").mkdir(exist_ok=True)
json.dump(summary, open("reports/prepare.json", "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
