"""승인 실행기의 검사 단위 테스트 (13-5). 이슈 본문은 에이전트가 쓴 것이라 믿지 않는다."""

import pytest

from src.actions import parse, validate


def body(plan: str) -> str:
    return f"## 제안\n되돌립니다\n\n```json\n{plan}\n```\n"


def test_deploy_is_allowed():
    assert validate(parse(body('{"action": "deploy", "version": "8"}'))) == ("deploy", {"version": "8"})


def test_restore_is_allowed():
    assert validate(parse(body('{"action": "restore", "version": "8"}'))) == ("restore", {"version": "8"})


@pytest.mark.parametrize("plan", [
    '{"action": "delete_model", "name": "flowers-classifier"}',          # 목록에 없는 조치
    '{"action": "deploy", "version": "8; rm -rf /"}',                    # 값 형식 위반 (명령 끼워 넣기)
    '{"action": "deploy", "version": "8", "force": "true"}',             # 모르는 인자
    '{"action": "silence", "alertname": "ServiceDown", "hours": "72"}',  # 24시간 초과
])
def test_rejected(plan):
    with pytest.raises(ValueError):
        validate(parse(body(plan)))


def test_two_json_blocks_rejected():                 # 사람이 본 것과 다른 블록이 실행되지 않게
    with pytest.raises(ValueError):
        parse(body('{"action": "retrain"}') + body('{"action": "deploy", "version": "1"}'))
