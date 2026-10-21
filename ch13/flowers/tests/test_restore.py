"""배포가 실패했을 때 별칭을 실행 전 상태로 되돌리는가 (07-5). MLflow 서버 없이 돈다."""

import pytest

from src.registry import restore_aliases


class FakeClient:
    """별칭만 흉내 내는 레지스트리."""

    def __init__(self, aliases: dict):
        self.aliases = dict(aliases)

    def get_registered_model(self, name):
        return type("RegisteredModel", (), {"aliases": dict(self.aliases)})()

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name, alias):
        del self.aliases[alias]


@pytest.mark.parametrize("before", [
    {"champion": "4"},                                         # previous가 아직 없음 (첫 승격 직후)
    {"champion": "4", "previous": "1"},
    {},                                                        # champion도 없던 첫 배포
    {"champion": "4", "previous": "1", "challenger": "7"},     # challenger는 건드리지 않는다
])
def test_failed_deploy_restores_aliases_exactly(before):
    client = FakeClient(before)
    if "champion" in before:                                   # deploy.py 2단계와 같은 변경
        client.set_registered_model_alias("flowers-classifier", "previous", before["champion"])
    client.set_registered_model_alias("flowers-classifier", "champion", "7")
    restore_aliases(client, before)
    assert client.aliases == before                            # 별칭 전체가 실행 전과 같다
