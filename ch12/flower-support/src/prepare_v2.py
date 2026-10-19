"""재학습용 데이터: 8장 학습 데이터 + 보강 데이터를 프롬프트 v2로 다시 만든다 (09-5).

모델은 학습 때 본 지시문에 맞춰진다(09-4). v2 프롬프트로 서빙할 모델이라면 v2로 학습한다.

입력: data/prepared/{train,val}.jsonl, data/augment/generated.jsonl, eval/evalset.jsonl
출력: data/prepared_v2/{train,val}.jsonl, reports/prepare_v2.json
"""

import json
import random
from pathlib import Path

import yaml

from src.build_evalset import ngrams, similarity
from src.prompts import VERSIONS
from src.schema import FIELDS

SYSTEM_V2 = VERSIONS[2][0]
seed = yaml.safe_load(open("params.yaml", encoding="utf-8"))["augment"]["seed"]


def example(ex_id, inquiry: str, target: dict) -> dict:
    return {"id": ex_id,
            "prompt": [{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": inquiry}],
            "completion": [{"role": "assistant", "content": json.dumps({k: target[k] for k in FIELDS}, ensure_ascii=False)}],
            "chat_template_kwargs": {"enable_thinking": False}}


def from_v1(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [example(r["id"], r["prompt"][1]["content"], json.loads(r["completion"][0]["content"])) for r in rows]


aug = [json.loads(line) for line in open("data/augment/generated.jsonl", encoding="utf-8")]
kept = [r for r in aug if all(r["check"][k] == r[k] for k in ("category", "urgency", "sentiment"))]

# 보강 데이터가 평가 세트와 겹치면 시험 문제를 가르친 셈이 된다
eval_grams = [ngrams(json.loads(line)["inquiry"]) for line in open("eval/evalset.jsonl", encoding="utf-8")]
sims = [max(similarity(ngrams(r["inquiry"]), g) for g in eval_grams) for r in kept]

rng = random.Random(seed)
rng.shuffle(kept)
n_val = len(kept) // 10
aug_val, aug_train = kept[:n_val], kept[n_val:]
train = from_v1("data/prepared/train.jsonl") + [example(r["id"], r["inquiry"], r) for r in aug_train]
val = from_v1("data/prepared/val.jsonl") + [example(r["id"], r["inquiry"], r) for r in aug_val]

Path("data/prepared_v2").mkdir(parents=True, exist_ok=True)
for name, items in (("train", train), ("val", val)):
    with open(f"data/prepared_v2/{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
report = {"augment_generated": len(aug), "augment_kept": len(kept), "n_train": len(train), "n_val": len(val),
          "augment_eval_similarity_max": round(max(sims), 3), "augment_eval_similarity_mean": round(sum(sims) / len(sims), 3)}
with open("reports/prepare_v2.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
