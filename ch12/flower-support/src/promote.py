"""레지스트리 별칭으로 배포 버전을 옮긴다: 승인(challenger → champion)과 롤백(previous → champion) (10-5).

모델 별칭과 프롬프트 별칭을 항상 함께 옮긴다. 프롬프트 버전은 모델 버전의 prompt_uri 태그(09-5의 게이트가 기록)를 따른다.
서빙 반영은 게이트웨이의 /reload가 한다.

실행:
    uv run python -m src.promote approve      # challenger를 champion으로 (지금 champion은 previous로)
    uv run python -m src.promote rollback     # previous를 champion으로 되돌린다
"""

import getpass
import sys
from datetime import datetime

import mlflow

MODEL, PROMPT = "flower-support-adapter", "flower-support-extract"
client = mlflow.MlflowClient()


def alias(name: str) -> str | None:
    return client.get_registered_model(MODEL).aliases.get(name)


def point(version: str):
    """champion을 version으로, production 프롬프트를 그 버전이 게이트에서 함께 평가된 프롬프트로."""
    client.set_registered_model_alias(MODEL, "champion", version)
    prompt_uri = client.get_model_version(MODEL, version).tags["prompt_uri"]          # prompts:/이름/버전
    mlflow.genai.set_prompt_alias(PROMPT, alias="production", version=int(prompt_uri.rsplit("/", 1)[1]))
    return prompt_uri


action = sys.argv[1]
before = alias("champion")
if action == "approve":
    target = alias("challenger")
    if not target:
        sys.exit("challenger가 없습니다. 게이트(09-5)를 먼저 통과시키세요.")
    client.set_registered_model_alias(MODEL, "previous", before)
    prompt_uri = point(target)
    client.delete_registered_model_alias(MODEL, "challenger")
    client.set_model_version_tag(MODEL, target, "approved_by", getpass.getuser())
    client.set_model_version_tag(MODEL, target, "approved_at", datetime.now().isoformat(timespec="seconds"))
elif action == "rollback":
    target = alias("previous")
    if not target:
        sys.exit("previous가 없습니다. 되돌릴 버전이 없습니다.")
    prompt_uri = point(target)
    client.delete_registered_model_alias(MODEL, "previous")
    client.set_model_version_tag(MODEL, before, "rolled_back_at", datetime.now().isoformat(timespec="seconds"))
else:
    sys.exit("approve 또는 rollback")
print(f"{action}: champion v{before} → v{target}, 프롬프트 production → {prompt_uri}")
print("서빙에 반영하려면 게이트웨이에 /reload를 요청하세요.")
