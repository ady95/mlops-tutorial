"""검토 기록을 적용해 평가 세트를 확정하고, 학습 데이터와 겹치지 않는지 검사한다 (09-2).

실행: python src/build_evalset.py evalset | holdout
입력: eval/raw/<이름>_generated.jsonl, eval/review_<이름>.yaml, data/prepared/train.jsonl
출력: eval/<이름>.jsonl, reports/<이름>.json
"""

import json
from collections import Counter
from pathlib import Path

import yaml

FIELDS = ["category", "urgency", "sentiment", "order_id", "product"]


def ngrams(text: str, n: int = 5) -> set[str]:
    text = "".join(text.split())
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def main(name: str):
    rows = [json.loads(line) for line in open(f"eval/raw/{name}_generated.jsonl", encoding="utf-8")]
    review = yaml.safe_load(open(f"eval/review_{name}.yaml", encoding="utf-8"))
    disagreed = {r["id"] for r in rows if any(r["check"][k] != r[k] for k in ("category", "urgency", "sentiment"))}
    reviewed = set(review["changes"]) | set(review["kept"])
    assert disagreed == reviewed, f"검토하지 않은 불일치: {sorted(disagreed - reviewed)}, 불필요한 검토: {sorted(reviewed - disagreed)}"

    # 학습 데이터와 글자 5-gram이 많이 겹치는 문의가 있으면 평가가 아니라 암기 확인이 된다
    train = [json.loads(line)["prompt"][1]["content"] for line in open("data/prepared/train.jsonl", encoding="utf-8")]
    train_grams = [ngrams(t) for t in train]

    out, sims = [], []
    for r in rows:
        gold = {k: r[k] for k in FIELDS}
        gold.update({k: v for k, v in review["changes"].get(r["id"], {}).items() if k != "reason"})
        grams = ngrams(r["inquiry"])
        sim = max(similarity(grams, g) for g in train_grams)
        sims.append(sim)
        out.append({"id": r["id"], "slice": r["slice"], "inquiry": r["inquiry"], "gold": gold,
                    "reference_summary": r["summary"], "max_train_similarity": round(sim, 3)})

    with open(f"eval/{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 다양성 점검: 처음 만든 세트는 함정 20건 중 19건이 같은 번호(FL-12345)였다
    traps = [r["trap_id"] for r in rows if r["slice"] == "주문번호 함정"]
    assert len(set(traps)) == len(traps), f"함정 번호 중복: {Counter(traps).most_common(3)}"
    report = {"n": len(out), "slices": dict(Counter(r["slice"] for r in out)), "trap_distinct": len(set(traps)),
              "disagreed": len(disagreed), "changed": len(review["changes"]),
              "train_similarity_max": round(max(sims), 3), "train_similarity_mean": round(sum(sims) / len(sims), 3),
              "over_0.5": sum(s > 0.5 for s in sims)}
    if name == "holdout":   # 보류 세트는 개발용 평가 세트·보강 데이터와도 겹치지 않아야 한다 (09-5)
        others = [json.loads(line)["inquiry"] for line in open("eval/evalset.jsonl", encoding="utf-8")]
        others += [json.loads(line)["inquiry"] for line in open("data/augment/generated.jsonl", encoding="utf-8")]
        other_grams = [ngrams(t) for t in others]
        report["dev_augment_similarity_max"] = round(
            max(max(similarity(ngrams(r["inquiry"]), g) for g in other_grams) for r in out), 3)
    Path("reports").mkdir(exist_ok=True)
    with open(f"reports/{name}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
