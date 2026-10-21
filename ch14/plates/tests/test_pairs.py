"""조합 확인 (14-2). 판정을 통과한 (검출, 인식) 조합만 배포·롤백할 수 있는가 — 레지스트리를 흉내 내어 돈다."""

from types import SimpleNamespace as NS

import pytest

from src.deploy import check_pair
from src.registry import verified_pairs


class FakeClient:
    def __init__(self, versions: dict, champions: dict):
        self.versions, self.champions = versions, champions   # 이름 → [(버전, 태그)], 이름 → champion

    def search_model_versions(self, query):
        name = query.split("'")[1]
        return [NS(version=v, tags=t) for v, t in self.versions.get(name, [])]

    def get_model_version_by_alias(self, name, alias):
        from mlflow.exceptions import MlflowException
        if name not in self.champions:
            raise MlflowException("no alias")
        return NS(version=self.champions[name])


def gate(partner: str, passed: bool = True) -> dict:
    return {"gate_partner": partner, "gate_passed": str(passed)}


# 14-2의 실측 상태: (1,1) (1,2) (2,2)는 통과, 검출 v2 + 인식 v1은 판정 불통과 뒤 다시 짝 v2와 통과
VERSIONS = {"plate-detector": [("1", gate("recognizer v1")), ("2", gate("recognizer v2"))],
            "plate-recognizer": [("1", gate("detector v1")), ("2", gate("detector v1"))]}


def test_verified_pairs():
    assert verified_pairs(FakeClient(VERSIONS, {})) == {("1", "1"), ("1", "2"), ("2", "2")}


@pytest.mark.parametrize("kind, new, champions, allowed", [
    ("recognizer", "1", {"plate-detector": "2"}, False),   # d2 + r1: 판정한 적 없음 (14-2의 롤백 거부)
    ("detector", "1", {"plate-recognizer": "2"}, True),    # d1 + r2: 판정 통과
    ("recognizer", "1", {"plate-detector": "1"}, True),    # d1 + r1
    ("detector", "1", {}, True),                           # 맨 처음: 짝 champion이 없으면 확인하지 않는다
])
def test_check_pair(monkeypatch, kind, new, champions, allowed):
    client = FakeClient(VERSIONS, champions)
    if allowed:
        check_pair(client, kind, new)
    else:
        with pytest.raises(SystemExit):
            check_pair(client, kind, new)
