"""champion 별칭을 previous 버전으로 되돌린다 (05-5).

실행:
    uv run python src/rollback.py "롤백 사유"

배포 승인 기준을 다시 거치지 않는다. 이미 운영된 적이 있는 이전 버전으로 즉시 돌아가는 비상 절차다.
"""

import getpass
import sys
import time
from datetime import datetime

import mlflow

NAME = "flowers-classifier"
reason = sys.argv[1] if len(sys.argv) > 1 else "(사유 없음)"
client = mlflow.MlflowClient()

start = time.perf_counter()
bad = client.get_model_version_by_alias(NAME, "champion")
prev = client.get_model_version_by_alias(NAME, "previous")

client.set_registered_model_alias(NAME, "champion", prev.version)
client.delete_registered_model_alias(NAME, "previous")
stamp = datetime.now().isoformat(timespec="seconds")
client.set_model_version_tag(NAME, bad.version, "rolled_back", f"{stamp} {getpass.getuser()}: {reason}")
elapsed = time.perf_counter() - start

print(f"롤백: champion v{bad.version} → v{prev.version} ({elapsed:.2f}초)")
print(f"v{bad.version}에 rolled_back 태그를 남겼습니다: {reason}")
