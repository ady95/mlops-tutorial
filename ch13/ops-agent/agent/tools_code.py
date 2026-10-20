"""코딩 도구 (13-1). 작업 폴더(샌드박스) 안에서만 읽고 쓰고, 테스트를 돌린다.

에이전트에게 실제 저장소를 주지 않는다. 복사본에서 일하게 하고, 결과는 사람이 diff로 검토한 뒤 반영한다.
"""

import os
import subprocess
from pathlib import Path

from agent.core import tool

ROOT = Path(os.environ.get("WORKSPACE", "workspace")).resolve()
PROTECTED = (".git", ".github", ".venv", "data")     # 읽기도 쓰기도 안 되는 곳 — 워크플로·데이터는 사람이 고친다


def inside(path: str) -> Path:
    p = (ROOT / path).resolve()
    if ROOT not in p.parents and p != ROOT:
        raise PermissionError(f"작업 폴더 밖: {path}")
    if any(part in PROTECTED for part in p.relative_to(ROOT).parts):
        raise PermissionError(f"보호된 경로: {path}")
    return p


@tool("작업 폴더의 파일 목록 (보호된 폴더 제외).", path={"type": "string"})
def list_files(path: str = ".") -> list:
    base = inside(path)
    return sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in base.rglob("*")
                  if p.is_file() and not any(x in PROTECTED or x == "__pycache__" for x in p.relative_to(ROOT).parts))


@tool("파일 내용을 읽는다.", path={"type": "string"})
def read_file(path: str) -> str:
    return inside(path).read_text(encoding="utf-8")


@tool("파일 전체를 새 내용으로 쓴다(없으면 만든다). 고칠 때도 파일 전체를 다시 쓴다.",
      path={"type": "string"}, content={"type": "string"})
def write_file(path: str, content: str) -> str:
    p = inside(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    return f"썼음: {path} ({len(content.splitlines())}줄)"


def test_env() -> dict:
    """테스트는 작업 폴더 프로젝트의 환경에서 돈다. TEST_VENV를 주면 그 가상환경을 함께 쓴다(매번 새로 만들지 않게).
    이 설정은 테스트 프로세스에만 준다 — 에이전트 자신의 환경과 섞이면 패키지가 서로 덮어쓴다 (13-1 실측)."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    if "TEST_VENV" in os.environ:
        env["UV_PROJECT_ENVIRONMENT"] = os.environ["TEST_VENV"]
    return env


def pytest(*paths: str) -> subprocess.CompletedProcess:
    cmd = os.environ.get("TEST_COMMAND", "uv run --frozen --extra cpu pytest -q").split() + list(paths)
    return subprocess.run(cmd, cwd=ROOT, env=test_env(), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=900)


@tool("단위 테스트(pytest)를 돌리고 결과 끝부분을 돌려준다.")
def run_tests() -> str:
    proc = pytest()
    return f"종료 코드 {proc.returncode}\n" + "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])


@tool("파일의 한 부분을 바꾼다. old는 파일 안에 정확히 한 번 있어야 하고(공백·줄바꿈까지), new로 바뀐다.",
      path={"type": "string"}, old={"type": "string"}, new={"type": "string"})
def replace_in_file(path: str, old: str, new: str) -> str:
    p = inside(path)
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if n != 1:
        return f"바꾸지 않음: old가 {n}번 나옴 (정확히 한 번이어야 함). read_file로 다시 확인하세요."
    p.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    return f"바꿨음: {path}"


@tool("새 파일을 만든다. 이미 있는 파일이면 거절한다(고칠 때는 replace_in_file).",
      path={"type": "string"}, content={"type": "string"})
def create_file(path: str, content: str) -> str:
    p = inside(path)
    if p.exists():
        return f"이미 있음: {path}. 고치려면 replace_in_file을 쓰세요."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    return f"만들었음: {path}"


CODE_TOOLS = [list_files, read_file, write_file, run_tests]              # 파일 전체를 다시 쓰는 도구
EDIT_TOOLS = [list_files, read_file, replace_in_file, create_file, run_tests]   # 부분만 바꾸는 도구 (13-1 비교)
