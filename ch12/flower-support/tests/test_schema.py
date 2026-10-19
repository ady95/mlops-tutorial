"""출력 파싱과 채점 단위 테스트 (08-1). GPU·모델 없이 돈다."""

from src.schema import FIELDS, SYSTEM_PROMPT, parse_output, score

GOLD = {"category": "배송", "urgency": "높음", "sentiment": "부정", "order_id": "FL-000123",
        "product": "장미 꽃다발", "summary": "배송 지연 문의"}


def test_parse_plain_and_fenced_json():
    text = '{"category": "배송"}'
    assert parse_output(text) == {"category": "배송"}
    assert parse_output("```json\n" + text + "\n```") == {"category": "배송"}


def test_parse_rejects_non_object():
    assert parse_output("배송 문의입니다") is None
    assert parse_output("[1, 2]") is None


def test_perfect_prediction():
    result = score(dict(GOLD, summary="다른 요약이어도 됨"), GOLD)
    assert all(result.values())


def test_wrong_label_and_missing_key():
    pred = {k: v for k, v in GOLD.items() if k != "product"} | {"urgency": "급함"}
    result = score(pred, GOLD)
    assert result["json_valid"] and not result["schema_valid"]
    assert not result["urgency"] and not result["product"] and result["category"]


def test_invalid_json_scores_zero():
    result = score(None, GOLD)
    assert not any(result.values())


def test_prompt_lists_every_field():
    assert all(f in SYSTEM_PROMPT for f in FIELDS)
