"""승인 실행기의 restore 전체 경로 (13-5). 레지스트리와 추론 서버를 흉내 내어 서버 없이 돈다.

reload 연결 실패·시간 초과·HTTP 오류·버전 불일치 어느 것이든, 별칭은 실행 전과 같아야 하고 실패로 끝나야 한다.
"""

import httpx
import mlflow
import pytest

from src import actions


class FakeRegistry:
    def __init__(self, aliases: dict):
        self.aliases = dict(aliases)

    def get_model_version(self, name, version):
        return type("MV", (), {"tags": {"approved_by": "someone"}})()

    def get_registered_model(self, name):
        return type("RM", (), {"aliases": dict(self.aliases)})()

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name, alias):
        del self.aliases[alias]

    def set_model_version_tag(self, *args):
        pass


class FakeServer:
    """mode: ok(reload하면 champion을 불러옴) / stuck(reload해도 옛 버전) / connect / timeout / http500"""

    def __init__(self, registry: FakeRegistry, mode: str):
        self.registry, self.mode, self.served = registry, mode, registry.aliases["champion"]

    def post(self, url, **kwargs):
        request = httpx.Request("POST", url)
        if self.mode == "connect":
            raise httpx.ConnectError("connection refused", request=request)
        if self.mode == "http500":
            self.served = self.registry.aliases["champion"]      # 서버는 바뀌었는데 500을 돌려준 경우
            return httpx.Response(500, request=request)
        if self.mode != "stuck":
            self.served = self.registry.aliases["champion"]
        return httpx.Response(200, request=request, json={"after": self.served})

    def get(self, url, **kwargs):
        request = httpx.Request("GET", url)
        if self.mode == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, request=request, json={"status": "ok", "model_version": self.served})


@pytest.fixture
def setup(monkeypatch):
    def make(before: dict, mode: str):
        registry = FakeRegistry(before)
        server = FakeServer(registry, mode)
        monkeypatch.setattr(mlflow, "MlflowClient", lambda: registry)
        monkeypatch.setattr(actions.httpx, "post", server.post)
        monkeypatch.setattr(actions.httpx, "get", server.get)
        monkeypatch.setenv("API_URL", "http://api")
        monkeypatch.setenv("RELOAD_TOKEN", "t")
        return registry
    return make


BEFORE = [{"champion": "4"}, {"champion": "4", "previous": "1"}]


@pytest.mark.parametrize("before", BEFORE)
def test_restore_success(setup, before):
    registry = setup(before, "ok")
    assert actions.restore("7").endswith("(종료 코드 0)")
    assert registry.aliases == {"champion": "7", "previous": "4"}


@pytest.mark.parametrize("before", BEFORE)
@pytest.mark.parametrize("mode", ["stuck", "connect", "timeout", "http500"])
def test_restore_failure_puts_aliases_back(setup, before, mode):
    registry = setup(before, mode)
    message = actions.restore("7")
    assert message.endswith("(종료 코드 1)"), message
    assert registry.aliases == before          # 별칭 전체가 실행 전과 같다
