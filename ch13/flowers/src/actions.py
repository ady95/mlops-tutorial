"""에이전트 제안의 실행기 (13-5). 사람이 이슈에 /approve를 달면 approve 워크플로가 부른다.

실행할 수 있는 일은 ACTIONS에 적은 것뿐이다. 에이전트는 무엇을 할지 '제안'만 하고, 권한은 이 파일과 워크플로가 쥔다.
이슈 본문은 에이전트가 썼으므로 믿지 않는다: JSON 블록 하나만 읽고, 조치 이름·인자 이름·값 형식을 모두 검사한다.

    uv run python -m src.actions          # 환경변수: GITHUB_TOKEN, GITHUB_REPOSITORY, ISSUE, APPROVER
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import httpx

ACTIONS = {                                          # 조치 → 인자 이름 → 값 형식
    "deploy": {"version": r"^\d{1,4}$"},             # 7장 deploy.py 그대로 (판정 재확인, 실패하면 되돌림)
    "restore": {"version": r"^\d{1,4}$"},            # 승인을 거쳐 운영된 적 있는 버전으로 되돌리기 (판정 없이)
    "silence": {"alertname": r"^[A-Za-z]{3,40}$", "hours": r"^([1-9]|1\d|2[0-4])$"},   # 최대 24시간
    "retrain": {},                                   # 학습 파이프라인 워크플로 실행
}
API = f"https://api.github.com/repos/{os.environ.get('GITHUB_REPOSITORY', '')}"
HEADERS = {"Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}", "Accept": "application/vnd.github+json"}


def parse(body: str) -> dict:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", body or "", flags=re.S)
    if len(blocks) != 1:
        raise ValueError(f"JSON 블록이 하나여야 함 ({len(blocks)}개)")
    return json.loads(blocks[0])


def validate(plan: dict) -> tuple[str, dict]:
    action = plan.get("action")
    if action not in ACTIONS:
        raise ValueError(f"허용되지 않은 조치: {action}")
    args = {k: str(v) for k, v in plan.items() if k != "action"}
    if sorted(args) != sorted(ACTIONS[action]):
        raise ValueError(f"{action}의 인자는 {sorted(ACTIONS[action])}여야 함: {sorted(args)}")
    for k, pattern in ACTIONS[action].items():
        if not re.fullmatch(pattern, args[k]):
            raise ValueError(f"{k} 값 형식이 맞지 않음: {args[k]!r}")
    return action, args


def execute(action: str, args: dict) -> str:
    if action == "deploy":
        p = subprocess.run([sys.executable, "-m", "src.deploy", args["version"]], capture_output=True, text=True)
        return (p.stdout + p.stderr)[-3000:] + f"\n(종료 코드 {p.returncode})"
    if action == "restore":
        return restore(args["version"])
    if action == "silence":
        now = datetime.now(timezone.utc)
        body = {"matchers": [{"name": "alertname", "value": args["alertname"], "isRegex": False}],
                "startsAt": now.isoformat(), "endsAt": (now + timedelta(hours=int(args["hours"]))).isoformat(),
                "createdBy": os.environ.get("APPROVER", "approver"), "comment": f"승인된 에이전트 제안 #{os.environ['ISSUE']}"}
        r = httpx.post(os.environ.get("ALERTMANAGER_URL", "http://localhost:9093") + "/api/v2/silences", json=body, timeout=10)
        r.raise_for_status()
        return f"묵음 {args['hours']}시간: {args['alertname']} (id {r.json()['silenceID'][:8]})"
    if action == "retrain":
        r = httpx.post(f"{API}/actions/workflows/pipeline.yml/dispatches", headers=HEADERS, json={"ref": "main"}, timeout=30)
        r.raise_for_status()
        return "pipeline 워크플로를 실행했음 — 결과는 Actions에서 확인"


def restore(version: str) -> str:
    """승인을 거쳐 운영된 적 있는(approved_by 태그가 있는) 버전으로 되돌린다. 판정은 다시 하지 않는다 —
    deploy.py는 '판정 때의 champion'이 바뀌면 멈추므로, 누군가 champion을 잘못 바꾼 사고에서는 되돌릴 수 없다 (13-5 실측)."""
    import mlflow
    name = "flowers-classifier"
    client = mlflow.MlflowClient()
    if "approved_by" not in client.get_model_version(name, version).tags:
        return f"되돌리지 않음: v{version}은 승인을 거쳐 운영된 적이 없음 (deploy로 판정부터)\n(종료 코드 1)"
    current = client.get_model_version_by_alias(name, "champion").version
    client.set_registered_model_alias(name, "champion", version)
    client.set_registered_model_alias(name, "previous", current)
    api, token = os.environ["API_URL"], os.environ["RELOAD_TOKEN"]
    r = httpx.post(f"{api}/reload", headers={"X-Reload-Token": token}, timeout=120)
    served = httpx.get(f"{api}/health", timeout=10).json()["model_version"]
    if served != version:
        client.set_registered_model_alias(name, "champion", current)
        return f"서버가 v{served}로 응답 — 별칭을 v{current}로 되돌림\n(종료 코드 1)"
    client.set_model_version_tag(name, version, "restored_by", os.environ.get("APPROVER", "approver"))
    return f"champion v{current} → v{version} 되돌림, reload {r.json()} → 서버 응답 v{served}\n(종료 코드 0)"


def comment(text: str):
    httpx.post(f"{API}/issues/{os.environ['ISSUE']}/comments", headers=HEADERS, json={"body": text}, timeout=30)


def main():
    issue = httpx.get(f"{API}/issues/{os.environ['ISSUE']}", headers=HEADERS, timeout=30).json()
    try:
        action, args = validate(parse(issue["body"]))
    except ValueError as e:
        comment(f"실행하지 않음: {e}")
        sys.exit(1)
    print(f"승인자 {os.environ.get('APPROVER')}: {action} {args}")
    result = execute(action, args)
    ok = "(종료 코드 0)" in result or action not in ("deploy", "restore")
    comment(f"{'실행 완료' if ok else '실행 실패'} — 승인자 @{os.environ.get('APPROVER')}\n\n```text\n{result}\n```")
    if ok:
        httpx.patch(f"{API}/issues/{os.environ['ISSUE']}", headers=HEADERS, json={"state": "closed"}, timeout=30)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
