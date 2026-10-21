"""계보 조회·점검 (14-1). MLflow 서버 없이, 레지스트리를 흉내 내어 돈다."""

from types import SimpleNamespace as NS

import pytest

import lineage

RUN_TAGS = {"git_commit": "a" * 40, "git_dirty": "False", "data_prepared_md5": "6f760de4"}


class FakeClient:
    def __init__(self, versions: dict, aliases: dict, runs: dict):
        self.versions, self.aliases, self.runs = versions, aliases, runs

    def get_model_version(self, name, version):
        return self.versions[(name, str(version))]

    def get_registered_model(self, name):
        return NS(name=name, aliases=self.aliases.get(name, {}))

    def search_registered_models(self):
        return [self.get_registered_model(n) for n in self.aliases]

    def get_run(self, run_id):
        return NS(info=NS(experiment_id="1"), data=NS(params={}, tags=self.runs[run_id]))

    def get_experiment(self, experiment_id):
        return NS(name="ch07-pipeline")


def mv(name, version, run_id=None, **tags):
    return NS(name=name, version=str(version), run_id=run_id, tags=tags)


def registry(onnx_source="9", champion="9", run_tags=RUN_TAGS, approved=True):
    approval = {"approved_by": "someone"} if approved else {}
    versions = {("flowers-classifier", "9"): mv("flowers-classifier", 9, "r9", test_f1_macro="0.9221", **approval),
                ("flowers-classifier", "4"): mv("flowers-classifier", 4, "r4", test_f1_macro="0.9259", **approval),
                ("flowers-classifier-onnx", "2"): mv("flowers-classifier-onnx", 2, None,
                                                     source_model="flowers-classifier", source_version=onnx_source)}
    aliases = {"flowers-classifier": {"champion": champion}, "flowers-classifier-onnx": {"champion": "2"}}
    return FakeClient(versions, aliases, {"r9": run_tags, "r4": run_tags})


@pytest.fixture
def use(monkeypatch):
    def install(client):
        monkeypatch.setattr(lineage, "client", client)
        return client
    return install


def test_answers_from_run_and_version_tags(use):
    client = use(registry())
    ans = lineage.answers(client.get_model_version("flowers-classifier", "9"))
    assert set(ans) == {"code", "clean", "data", "evaluation", "approval"}


def test_converted_model_inherits_code_and_data_but_not_approval(use):
    client = use(registry())
    ans = lineage.answers(client.get_model_version("flowers-classifier-onnx", "2"))
    assert ans["code"][3].startswith("← flowers-classifier v9") and ans["data"][3]
    assert "approval" not in ans                    # 운영에 올린 결정은 버전마다 따로


def test_stale_source_detected(use):
    client = use(registry(onnx_source="4", champion="9"))
    assert "원본의 지금 champion은 v9" in lineage.stale_source(client.get_model_version("flowers-classifier-onnx", "2"))
    use(registry(onnx_source="9", champion="9"))
    assert lineage.stale_source(lineage.client.get_model_version("flowers-classifier-onnx", "2")) == ""


def test_dirty_commit_is_mismatch(use):
    client = use(registry(run_tags=RUN_TAGS | {"git_dirty": "True"}))
    assert lineage.answers(client.get_model_version("flowers-classifier", "9"))["clean"][2] is False


@pytest.mark.parametrize("kwargs", [{}, {"approved": False}, {"run_tags": RUN_TAGS | {"git_dirty": "True"}}])
def test_audit_fails_while_onnx_has_no_approval(use, kwargs):
    use(registry(**kwargs))
    assert lineage.audit() == 1                     # ONNX 변환본은 승인이 없으므로 언제나 실패


def test_audit_passes_without_converted_model(use):
    client = use(registry())
    del client.aliases["flowers-classifier-onnx"]
    assert lineage.audit() == 0


def test_approver_is_masked():
    assert lineage.shown("approved_by", "someone") == "<승인자>"
