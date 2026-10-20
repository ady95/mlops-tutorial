"""파이프라인 코드를 고치는 코딩 에이전트 (13-1).

    uv run python codegen.py --source ../../ch12/flowers --tasks cases/codegen/tasks.yaml --repeat 3

작업마다 원본을 복사한 샌드박스(git 저장소)에서 에이전트가 일하고, 끝나면
1) 에이전트가 쓴 테스트를 포함한 전체 테스트, 2) 에이전트가 보지 못한 인수 테스트(cases/codegen/<id>/),
3) 기존 테스트 파일을 고쳤는지, 4) 바꾼 파일 목록을 기록한다. 결과는 outputs/codegen/.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import mlflow
import yaml

SYSTEM = """너는 이 저장소에서 일하는 코딩 에이전트다. 사람의 작업 요청을 받아 코드를 고친다.

규칙:
1. 먼저 관련 파일을 읽고, 기존 코드의 방식(이름, 주석, 구조)을 따른다.
2. 요청한 것만 바꾼다. 관계없는 파일은 손대지 않는다.
3. 기존 테스트를 고치거나 지우지 않는다. 새 테스트는 새 함수로 더한다.
4. 바꾼 뒤 run_tests로 확인하고, 실패하면 원인을 찾아 고친다.
5. 끝나면 바꾼 파일과 이유를 요약한다. 확인하지 못한 것(예: MLflow 서버가 있어야 도는 코드)은 그렇다고 쓴다."""


def sh(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def make_workspace(source: Path, ws: Path):
    if ws.exists():   # 윈도에서 .git 객체 파일은 읽기 전용이라 그냥 지우면 PermissionError
        shutil.rmtree(ws, onexc=lambda fn, path, exc: (os.chmod(path, 0o700), fn(path)))
    shutil.copytree(source, ws, ignore=shutil.ignore_patterns(".venv", "__pycache__", "data", "outputs"))
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=agent", "-c", "user.email=agent@example.com", "commit", "-qm", "base"]):
        sh(cmd, ws)


def pytest(ws: Path, *paths: str) -> dict:
    from agent import tools_code
    tools_code.ROOT = ws
    p = tools_code.pytest(*paths)
    last = (p.stdout.strip().splitlines() or [""])[-1]
    return {"ok": p.returncode == 0, "summary": last}


def lost_comments(ws: Path) -> list[str]:
    """원래 있던 주석 중 사라진 것. 요청과 상관없는 곳을 건드렸는지 보는 간단한 신호."""
    diff = sh(["git", "diff"], ws).stdout.splitlines()
    removed = [line.split("#", 1)[1].strip() for line in diff if line.startswith("-") and not line.startswith("---") and "#" in line]
    added = "\n".join(line for line in diff if line.startswith("+"))
    return [c for c in removed if c and c not in added]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--only")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--first", type=int, default=1, help="회차 번호 시작 (중단된 실행 이어 하기)")
    ap.add_argument("--label", default="default")
    ap.add_argument("--tools", choices=["whole", "edit"], default="whole", help="파일 전체 쓰기 / 부분 바꾸기 (13-1 비교)")
    args = ap.parse_args()
    mlflow.set_experiment("ch13-ops-agent")
    tasks = yaml.safe_load(open(args.tasks, encoding="utf-8"))
    if args.only:
        tasks = [t for t in tasks if t["id"] in args.only.split(",")]
    out = Path("outputs/codegen") / args.label
    out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        for i in range(args.first, args.first + args.repeat):
            ws = (out / f"{t['id']}-{i}").resolve()
            make_workspace(Path(args.source), ws)
            os.environ["WORKSPACE"] = str(ws)
            from agent import tools_code            # 작업 폴더가 정해진 뒤에 불러온다
            tools_code.ROOT = ws
            from agent.core import run
            tools = tools_code.EDIT_TOOLS if args.tools == "edit" else tools_code.CODE_TOOLS
            r = run(t["request"], tools, SYSTEM, max_steps=30, name=f"codegen-{t['id']}")
            changed = sh(["git", "status", "--porcelain"], ws).stdout.split("\n")
            changed = sorted(line[3:] for line in changed if line.strip())
            old_tests_touched = [f for f in changed if f in ("tests/test_gate.py", "tests/test_training.py")]
            own = pytest(ws)
            shutil.copy(Path(args.tasks).parent / t["id"] / "test_acceptance.py", ws / "tests" / "test_acceptance.py")
            hidden = pytest(ws, "tests/test_acceptance.py")
            (ws.parent / f"{t['id']}-{i}.diff").write_text(sh(["git", "diff"], ws).stdout, encoding="utf-8")
            (ws.parent / f"{t['id']}-{i}.md").write_text(r["answer"] or "(끝내지 못함)", encoding="utf-8")
            row = {"task": t["id"], "run": i, "tools": args.tools, "steps": r["steps"], "tool_calls": len(r["calls"]),
                   "lost_comments": lost_comments(ws),
                   "run_tests_calls": sum(c["tool"] == "run_tests" for c in r["calls"]), "tokens": r["tokens"],
                   "seconds": r["seconds"], "changed": changed, "old_tests_touched": old_tests_touched,
                   "own_tests": own, "acceptance": hidden}
            print(json.dumps(row, ensure_ascii=False))
            (out / "summary.jsonl").open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
