"""champion 승격 → 추론 서버 reload → 응답 버전 확인. 확인에 실패하면 되돌린다 (07-5).

실행:
    uv run python -m src.deploy [version]      # version을 비우면 challenger 버전
환경변수: MLFLOW_TRACKING_URI, API_URL, RELOAD_TOKEN, (GitHub Actions에서는) GITHUB_ACTOR
"""

import os
import sys
import time
from datetime import datetime

import httpx
import mlflow
import yaml

from src.registry import NAME, champion_tags, compared_champion, judge, restore_aliases

client = mlflow.MlflowClient()
version = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else \
    client.get_model_version_by_alias(NAME, "challenger").version
api, token = os.environ["API_URL"], os.environ["RELOAD_TOKEN"]
approver = os.environ.get("GITHUB_ACTOR", os.environ.get("USER", "unknown"))


def reload_and_check(expected: str) -> bool:
    """서버에 reload를 요청하고 응답 버전을 확인한다. 요청 자체가 실패해도 예외 대신 False."""
    try:
        r = httpx.post(f"{api}/reload", headers={"X-Reload-Token": token}, timeout=120)
        r.raise_for_status()
        served = httpx.get(f"{api}/health", timeout=10).json()["model_version"]
    except httpx.HTTPError as e:
        print(f"  reload 실패: {e}")
        return False
    print(f"  reload {r.json()} → 서버 응답 버전 v{served}")
    return served == expected


# 1. 승인 기준을 지금 시점의 champion과 다시 비교한다 (학습 이후 champion이
# 바뀌었을 수 있다)
#    판정 때 같은 테스트 세트로 잰 champion 점수(champion_* 태그)로 비교한다. 그 사이 champion이 바뀌었으면 다시 판정해야 한다 (12-5)
champ_version, _ = champion_tags()
tags = client.get_model_version(NAME, version).tags
if champ_version in (None, version):
    champ = None
elif tags.get("champion_version") == champ_version:
    champ = compared_champion(tags)
else:
    sys.exit(f"판정 때의 champion(v{tags.get('champion_version')})과 지금 champion(v{champ_version})이 다릅니다. "
             "파이프라인을 다시 실행해 지금 champion과 비교하세요")
checks = judge(tags, champ, yaml.safe_load(open("params.yaml"))["gate"])
print(f"{NAME} v{version} 배포 (현재 champion v{champ_version}, 승인자 {approver})")
for text, ok in checks:
    print(f"  [{'OK' if ok else '불가'}] {text}")
if not all(ok for _, ok in checks):
    sys.exit("배포 중단: 승인 기준 불통과")

# 2. 별칭 변경 (실패하면 되돌릴 수 있게 지금 별칭 전체를 기억해 둔다)
start = time.perf_counter()
before = dict(client.get_registered_model(NAME).aliases)
if champ_version and champ_version != version:
    client.set_registered_model_alias(NAME, "previous", champ_version)
client.set_registered_model_alias(NAME, "champion", version)

# 3. 서버 반영과 확인. 성공해야 승인 기록을 남기고, 실패하면 이전 상태로 되돌린다
if reload_and_check(version):
    if client.get_registered_model(NAME).aliases.get("challenger") == version:
        client.delete_registered_model_alias(NAME, "challenger")
    client.set_model_version_tag(NAME, version, "approved_by", approver)
    client.set_model_version_tag(NAME, version, "approved_at", datetime.now().isoformat(timespec="seconds"))
    print(f"배포 완료: champion v{version} ({time.perf_counter() - start:.1f}초)")
else:
    print("서버가 새 버전으로 응답하지 않습니다. 레지스트리를 되돌립니다.")
    restore_aliases(client, before)              # 원래 없던 별칭은 지운다
    if not champ_version or not reload_and_check(champ_version):
        print("경고: 서버 상태를 직접 확인하세요 (/health).")
    sys.exit(1)
