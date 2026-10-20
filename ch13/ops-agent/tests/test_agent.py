"""에이전트의 안전장치 단위 테스트 (13장). 모델·서버 없이 돈다."""

import pytest

from agent import grounding, tools_code
from agent.sanitize import sanitize
from agent.tools_action import propose_action


def test_sanitize_hides_internal_info():
    text = "target 203.0.113.7:9400 down, path /home/alice/work, api_key=abc123, 010-1234-5678"
    out = sanitize(text)
    assert "203.0.113.7" not in out and "alice" not in out and "abc123" not in out and "1234-5678" not in out


def test_sandbox_blocks_outside_and_protected(tmp_path):
    tools_code.ROOT = tmp_path
    (tmp_path / "src").mkdir()
    assert tools_code.inside("src") == tmp_path / "src"
    for bad in ("../secret.txt", ".github/workflows/x.yml", "data/raw/a.jpg", ".git/config"):
        with pytest.raises(PermissionError):
            tools_code.inside(bad)


def test_replace_requires_exactly_one_match(tmp_path):
    tools_code.ROOT = tmp_path
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    assert "2번" in tools_code.replace_in_file.fn("a.py", "x = 1", "x = 2")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_grounding_flags_made_up_numbers():
    messages = [{"role": "tool", "content": '{"val_f1_macro": 0.90963, "latency": 8.5451}'}]
    result = grounding.check("F1 0.9096, 지연 8.55ms, 정확도 0.9312", messages)
    assert result["unsupported"] == ["0.9312"]


def test_proposal_rejects_unknown_action():
    out = propose_action.fn("delete_model", {"name": "x"}, "t", "r", "e", "k")
    assert out.startswith("제안할 수 없음")
