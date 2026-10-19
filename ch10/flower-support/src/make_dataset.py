"""LLM으로 학습 데이터 합성: 라벨을 먼저 뽑고, 그 라벨에 맞는 문의를 쓰게 한 뒤, 다른 호출로 검수한다 (08-1).

실행:
    export OPENAI_API_KEY=...           # 표준 OpenAI API (OPENAI_BASE_URL로 호환 서버 지정 가능)
    export OPENAI_MODEL=...             # 문의를 쓸 모델
    export OPENAI_CHECK_MODEL=...       # 검수할 모델 (비우면 OPENAI_MODEL)
    uv run --no-project --with openai --with pyyaml python src/make_dataset.py

이미 만든 id는 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 만든다.
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

import yaml
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from schema import CATEGORIES, PRODUCTS, SENTIMENT, SENTIMENT_GUIDE, URGENCY, URGENCY_GUIDE  # noqa: E402

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["data"]
OUT = Path("data/raw/generated.jsonl")
STYLES = ["정중한 존댓말", "짧고 무뚝뚝한 말투", "메신저처럼 가벼운 구어체", "사정을 길게 설명하는 말투",
          "오타와 띄어쓰기 실수가 조금 섞인 말투"]
TOPICS = {  # 유형마다 구체적인 사연 하나를 골라 쓰게 한다 (힌트를 다 주면 한 문의에 다 섞어 쓴다)
    "배송": ["배송 지연", "도착 예정 시간 확인", "배송 상태가 바뀌지 않음", "다른 곳으로 잘못 배송됨"],
    "주문변경": ["받는 날짜 변경", "받는 주소 변경", "리본 문구 변경", "상품 구성 변경"],
    "취소환불": ["주문 취소", "환불 요청", "환불이 아직 안 들어옴", "부분 취소"],
    "상품품질": ["꽃이 시들어 도착", "사진과 다른 상품", "화분·포장 파손", "꽃 수가 적음"],
    "결제": ["결제 오류", "중복 결제", "결제 수단 문의", "영수증·세금계산서 발급"],
    "회원계정": ["로그인이 안 됨", "비밀번호 재설정 메일이 안 옴", "적립금이 안 들어옴", "회원 정보 변경"],
    "기타": ["매장 위치 문의", "기업 대량 주문 상담", "직원·서비스 칭찬", "꽃 관리 방법 문의"],
}


def sample_labels(rng: random.Random, i: int) -> dict:
    category = rng.choice(CATEGORIES)
    with_order = category not in ("회원계정", "기타") and rng.random() < 0.7
    with_product = category not in ("회원계정", "결제") and rng.random() < (0.3 if category == "기타" else 0.6)
    return {
        "id": i,
        "category": category,
        "topic": rng.choice(TOPICS[category]),
        "urgency": rng.choice(URGENCY),
        "sentiment": rng.choices(SENTIMENT, weights=[1, 2, 3])[0],
        "order_id": f"FL-{rng.randint(0, 999999):06d}" if with_order else None,
        "product": rng.choice(PRODUCTS) if with_product else None,
        "style": rng.choice(STYLES),
    }


def write_prompt(lab: dict) -> str:
    return f"""꽃 배달 쇼핑몰에 고객이 남긴 문의 글을 한국어로 하나 지어 주세요.
- 문의 유형: {lab['category']} — 사연: {lab['topic']}
- 긴급도: {lab['urgency']} — {URGENCY_GUIDE[lab['urgency']]}
- 고객 감정: {lab['sentiment']} — {SENTIMENT_GUIDE[lab['sentiment']]}
- 주문번호: {lab['order_id'] or '언급하지 않음'}
- 상품: {lab['product'] or '특정 상품을 언급하지 않음'}
- 말투: {lab['style']}
규칙: 주문번호와 상품명은 위에 적힌 그대로 쓴다. 사람 이름·전화번호·자세한 주소는 쓰지 않는다.
문의 유형이나 긴급도를 단어로 직접 말하지 말고 내용으로 드러낸다. 1~5문장.
JSON으로 답하세요: {{"inquiry": "문의 글", "summary": "문의 내용을 한 문장으로 요약"}}"""


def check_prompt(inquiry: str) -> str:
    return f"""다음은 꽃 배달 쇼핑몰의 고객 문의입니다. 분류해 주세요.
문의: {inquiry}
긴급도 기준: {URGENCY_GUIDE}
감정 기준: {SENTIMENT_GUIDE}
JSON으로 답하세요: {{"category": {CATEGORIES} 중 하나, "urgency": {URGENCY} 중 하나, "sentiment": {SENTIMENT} 중 하나}}"""


async def ask(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                             response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)


async def make_one(client, sem, lab, gen_model, check_model):
    async with sem:
        for _ in range(3):
            try:
                gen = await ask(client, gen_model, write_prompt(lab))
                chk = await ask(client, check_model, check_prompt(gen["inquiry"]))
                return {**lab, "inquiry": gen["inquiry"], "summary": gen["summary"],
                        "check": {k: chk.get(k) for k in ("category", "urgency", "sentiment")}}
            except Exception as e:  # 일시적 오류는 세 번까지 다시 시도
                err = e
        print(f"  id {lab['id']} 실패: {err}")
        return None


async def main():
    gen_model = os.environ["OPENAI_MODEL"]
    check_model = os.environ.get("OPENAI_CHECK_MODEL") or gen_model
    rng = random.Random(params["seed"])
    labels = [sample_labels(rng, i) for i in range(params["n_generate"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    todo = [lab for lab in labels if lab["id"] not in done]
    print(f"생성 대상 {len(todo)}건 (이미 있음 {len(done)}건)")

    client, sem = AsyncOpenAI(), asyncio.Semaphore(params["concurrency"])
    with OUT.open("a", encoding="utf-8", newline="\n") as f:   # 윈도에서도 LF로 (OS마다 해시가 달라지지 않게)
        for n, coro in enumerate(asyncio.as_completed([make_one(client, sem, lab, gen_model, check_model)
                                                        for lab in todo]), 1):
            row = await coro
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}")


if __name__ == "__main__":
    asyncio.run(main())
