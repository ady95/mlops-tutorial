"""운영 중 응답을 지키는 규칙들: 민감정보 가리기, 출력 규칙 검사, 지시 주입 의심, 분류 확신도 (11-1, 11-4).

모두 모델이나 서버 없이 도는 순수 함수라 단위 테스트할 수 있다 (tests/test_guards.py).
"""

import math
import re

from schema import CATALOG, CATEGORIES, FIELDS, ORDER_ID, SENTIMENT, URGENCY

# 개인정보: 모델에 보내기 전과 기록에 남기기 전에 가린다 (주문번호 FL-123456은 가리지 않는다)
PII = [
    (re.compile(r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}"), "[전화번호]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[이메일]"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"), "[카드번호]"),
]
INJECTION = re.compile(r"(이전|위의?|앞의?)\s*(지시|명령|규칙)[를을은는]?\s*(무시|잊)|ignore (all )?(previous|prior) instructions"
                       r"|(category|urgency|sentiment)\s*[를을은는]?\s*\S+\s*(으로|로)\s*(적|바꿔|분류|처리)", re.IGNORECASE)


def mask_pii(text: str) -> tuple[str, list[str]]:
    """민감정보를 가린 글과, 가린 종류 목록."""
    found = []
    for pattern, label in PII:
        if pattern.search(text):
            found.append(label.strip("[]"))
            text = pattern.sub(label, text)
    return text, found


def injection_suspected(text: str) -> bool:
    return bool(INJECTION.search(text))


def check_output(inquiry: str, result: dict | None) -> dict:
    """정답 없이 볼 수 있는 출력 규칙. 하나라도 False면 사람이 확인할 후보다."""
    if not isinstance(result, dict):
        return {"json_valid": False}
    order_id, product = result.get("order_id"), result.get("product")
    return {
        "json_valid": True,
        "schema_valid": (set(result) == set(FIELDS) and result.get("category") in CATEGORIES
                         and result.get("urgency") in URGENCY and result.get("sentiment") in SENTIMENT),
        # 환각 의심: 문의에 없는 값을 적었는가 (형식이 틀린 주문번호도 여기서 걸린다)
        "order_id_grounded": order_id is None or (bool(ORDER_ID.fullmatch(order_id)) and order_id in inquiry),
        "product_in_inquiry": product is None or product in inquiry,
        # 계약 위반: 받는 쪽 시스템이 모르는 상품 — 환각일 수도, 카탈로그에 없는 새 상품일 수도 있다
        "product_in_catalog": product is None or product in CATALOG,
    }


def category_confidence(tokens: list[dict]) -> float | None:
    """vLLM이 돌려준 토큰별 logprob에서, category 값의 첫 토큰 확률을 꺼낸다.

    tokens: [{"token": "...", "logprob": -0.01}, ...] (chat completions의 logprobs.content)
    """
    text = ""
    for i, t in enumerate(tokens):
        text += t["token"]
        if re.search(r'"category"\s*:\s*"$', text) and i + 1 < len(tokens):
            return math.exp(tokens[i + 1]["logprob"])
    return None
