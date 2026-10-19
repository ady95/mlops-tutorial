"""꽃 배달 쇼핑몰 고객 문의 → JSON 추출 과제의 정의 (08-1).

데이터 생성·학습·평가가 모두 이 파일의 라벨 목록과 채점 함수를 쓴다.
"""

import json
import re

CATEGORIES = ["배송", "주문변경", "취소환불", "상품품질", "결제", "회원계정", "기타"]
URGENCY = ["높음", "보통", "낮음"]
SENTIMENT = ["긍정", "중립", "부정"]
PRODUCTS = ["장미 꽃다발", "해바라기 꽃다발", "튤립 꽃다발", "데이지 꽃바구니", "민들레 화분",
            "튤립 화분", "장미 정기구독", "계절꽃 정기구독"]
ORDER_ID = re.compile(r"FL-\d{6}")
FIELDS = ["category", "urgency", "sentiment", "order_id", "product", "summary"]

# 사람마다 다르게 판단하기 쉬운 두 필드는 기준을 글로 정해 둔다 (생성·검수·학습 모두 같은 기준)
URGENCY_GUIDE = {"높음": "오늘·내일 안에 처리되지 않으면 문제가 생김. 임박한 배송일·행사, 주문 불가 상태 등",
                 "보통": "며칠 안에 답이 필요하지만 당장 급하지는 않음",
                 "낮음": "급하지 않은 정보 요청·건의·칭찬, 또는 고객이 천천히 답해도 된다고 함"}
SENTIMENT_GUIDE = {"긍정": "감사·만족·기대가 드러남",
                   "중립": "감정 표현 없이 사실과 요청만 말함",
                   "부정": "불만·실망·짜증이 드러남"}

SYSTEM_PROMPT = f"""당신은 꽃 배달 쇼핑몰의 고객 문의를 정리하는 도우미입니다.
고객 문의를 읽고 아래 키를 가진 JSON 객체 하나만 출력하세요. 다른 말은 쓰지 마세요.
- category: {", ".join(CATEGORIES)} 중 하나
- urgency: {" / ".join(f"{k}({v})" for k, v in URGENCY_GUIDE.items())} 중 하나
- sentiment: {" / ".join(f"{k}({v})" for k, v in SENTIMENT_GUIDE.items())} 중 하나
- order_id: 문의에 적힌 주문번호(FL-숫자6자리). 없으면 null
- product: {", ".join(PRODUCTS)} 중 문의에 나온 상품. 없으면 null
- summary: 문의 내용을 한 문장으로 요약"""


def parse_output(text: str) -> dict | None:
    """모델 출력에서 JSON 객체를 꺼낸다. 형식이 틀리면 None."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score(pred: dict | None, gold: dict) -> dict:
    """필드별 정답 여부. summary는 자동 채점하지 않는다 (9장 LLM 심판)."""
    result = {"json_valid": pred is not None}
    pred = pred or {}
    result["schema_valid"] = (set(pred) == set(FIELDS) and pred.get("category") in CATEGORIES
                              and pred.get("urgency") in URGENCY and pred.get("sentiment") in SENTIMENT)
    for key in ("category", "urgency", "sentiment", "order_id", "product"):
        result[key] = pred.get(key) == gold[key]
    return result
