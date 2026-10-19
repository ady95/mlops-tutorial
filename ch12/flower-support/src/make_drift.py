"""운영 중 바뀐 상황을 흉내 낸 문의: 쇼핑몰이 새 상품을 내놓았다 (11-4, 11-5).

모델이 학습할 때 상품 목록에 없던 '선인장 화분', '프리저브드 장미 박스'에 관한 문의를 8장과 같은 방식(라벨 먼저)으로 만든다.

실행:
    export OPENAI_API_KEY=... OPENAI_MODEL=...
    uv run python src/make_drift.py                 # 합성 (API)
    uv run python src/make_drift.py --convert-only  # 받은 데이터를 평가 형식으로만
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from make_dataset import STYLES, TOPICS, ask, write_prompt  # noqa: E402
from schema import CATEGORIES, SENTIMENT, URGENCY  # noqa: E402

NEW_PRODUCTS = ["선인장 화분", "프리저브드 장미 박스"]
OUT = Path("eval/raw/drift_generated.jsonl")


def sample(rng: random.Random, i: int) -> dict:
    category = rng.choice([c for c in CATEGORIES if c != "회원계정"])
    return {"id": f"D{i:03d}", "category": category, "topic": rng.choice(TOPICS[category]),
            "urgency": rng.choice(URGENCY), "sentiment": rng.choices(SENTIMENT, weights=[1, 2, 3])[0],
            "order_id": f"FL-{rng.randint(0, 999999):06d}" if rng.random() < 0.6 else None,
            "product": rng.choice(NEW_PRODUCTS), "style": rng.choice(STYLES)}


def write_eval(rows: list[dict]):
    """평가용 형식(eval_served가 읽는 형식)으로도 남긴다. 다른 모델의 검수는 거치지 않았다."""
    with open("eval/drift.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            gold = {k: r[k] for k in ("category", "urgency", "sentiment", "order_id", "product")}
            f.write(json.dumps({"id": r["id"], "slice": "새 상품", "inquiry": r["inquiry"], "gold": gold,
                                "reference_summary": r["summary"]}, ensure_ascii=False) + "\n")


async def main():
    rng = random.Random(11)
    labels = [sample(rng, i) for i in range(80)]
    client, sem, model = AsyncOpenAI(), asyncio.Semaphore(16), os.environ["OPENAI_MODEL"]

    async def one(lab):
        async with sem:
            gen = await ask(client, model, write_prompt(lab))
            return {**lab, "inquiry": gen["inquiry"], "summary": gen["summary"]}

    rows = await asyncio.gather(*(one(lab) for lab in labels))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_eval(rows)
    print(f"{len(rows)}건 → {OUT}, eval/drift.jsonl, 상품명이 그대로 들어간 문의 {sum(r['product'] in r['inquiry'] for r in rows)}건")


if __name__ == "__main__":
    if sys.argv[1:] == ["--convert-only"]:      # 받은 데이터(dataset/drift_generated.jsonl)를 eval/raw/에 두었을 때
        write_eval([json.loads(line) for line in OUT.open(encoding="utf-8")])
    else:
        asyncio.run(main())
