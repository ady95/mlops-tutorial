"""모델 하나만 배포·롤백한다 (14-2). 짝 모델은 건드리지 않는다.

    uv run python -m src.deploy deploy recognizer          # challenger → champion, 서버에 그 모델만 reload
    uv run python -m src.deploy rollback recognizer        # champion ← previous, 서버에 그 모델만 reload
    uv run python -m src.deploy deploy detector --no-reload   # 맨 처음, 서버를 띄우기 전
환경변수: MLFLOW_TRACKING_URI, API_URL(기본 http://localhost:8010), RELOAD_TOKEN

배포·롤백 모두 바꾼 뒤의 (검출, 인식) 조합이 판정을 통과한 적 있는 조합인지 먼저 확인한다.
모델 하나만 되돌려도 처음 보는 조합이 생길 수 있다 — 그 조합의 끝단 정확도는 아무도 재지 않았다.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import httpx
import mlflow

from src.evaluate import NAMES
from src.registry import OTHER, alias_version, verified_pairs

API = os.environ.get("API_URL", "http://localhost:8010")
APPROVER = os.environ.get("GITHUB_ACTOR", os.environ.get("USER", "unknown"))


def reload_and_check(kind: str, expected: str) -> bool:
    try:
        r = httpx.post(f"{API}/reload", params={"model": kind},
                       headers={"X-Reload-Token": os.environ["RELOAD_TOKEN"]}, timeout=120)
        r.raise_for_status()
        served = httpx.get(f"{API}/health", timeout=10).json()[kind]
    except httpx.HTTPError as e:
        print(f"  reload 실패: {e}")
        return False
    print(f"  reload {r.json()} → 서버의 {kind} v{served}")
    return served == expected


def check_pair(client, kind: str, new: str) -> None:
    partner = alias_version(client, OTHER[kind], "champion")
    # 맨 처음, 짝 champion이 아직 없을 때
    if partner is None:
        return
    pair = {kind: new, OTHER[kind]: partner}
    if (pair["detector"], pair["recognizer"]) not in verified_pairs(client):
        sys.exit(f"검출 v{pair['detector']} + 인식 v{pair['recognizer']}은 판정을 통과한 적 없는 조합입니다. "
                 "이 조합으로 다시 판정하거나, 짝 모델을 먼저 바꾸세요")


def switch(kind: str, new: str, old: str | None, reload: bool) -> None:
    client = mlflow.MlflowClient()
    check_pair(client, kind, new)
    start = time.perf_counter()
    before = {a: alias_version(client, kind, a) for a in ("champion", "previous")}   # 실패하면 이 상태로
    if old:
        client.set_registered_model_alias(NAMES[kind], "previous", old)
    client.set_registered_model_alias(NAMES[kind], "champion", new)
    if not reload or reload_and_check(kind, new):
        client.set_model_version_tag(NAMES[kind], new, "approved_by", APPROVER)
        client.set_model_version_tag(NAMES[kind], new, "approved_at", datetime.now().isoformat(timespec="seconds"))
        print(f"{NAMES[kind]} champion v{old} → v{new} ({time.perf_counter() - start:.1f}초)")
        return
    print("서버가 새 버전으로 응답하지 않습니다. 레지스트리를 되돌립니다.")
    for alias, version in before.items():          # 원래 없던 별칭은 지운다
        if version:
            client.set_registered_model_alias(NAMES[kind], alias, version)
        elif alias_version(client, kind, alias):
            client.delete_registered_model_alias(NAMES[kind], alias)
    if not old or not reload_and_check(kind, old):
        print("경고: 서버 상태를 직접 확인하세요 (/health).")
    sys.exit(1)


def deploy(kind: str, version: str | None, reload: bool) -> None:
    client = mlflow.MlflowClient()
    version = version or alias_version(client, kind, "challenger")
    if version is None:
        sys.exit(f"{NAMES[kind]}에 challenger가 없습니다")
    switch(kind, version, alias_version(client, kind, "champion"), reload)
    if alias_version(client, kind, "challenger") == version:
        client.delete_registered_model_alias(NAMES[kind], "challenger")


def rollback(kind: str, reload: bool) -> None:
    client = mlflow.MlflowClient()
    previous = alias_version(client, kind, "previous")
    if previous is None:
        sys.exit(f"{NAMES[kind]}에 previous가 없습니다")
    switch(kind, previous, alias_version(client, kind, "champion"), reload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["deploy", "rollback"])
    p.add_argument("kind", choices=list(NAMES))
    p.add_argument("version", nargs="?")
    p.add_argument("--no-reload", action="store_true")
    args = p.parse_args()
    if args.cmd == "deploy":
        deploy(args.kind, args.version, not args.no_reload)
    else:
        rollback(args.kind, not args.no_reload)


if __name__ == "__main__":
    main()
