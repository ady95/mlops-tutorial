"""배포 승인 기준(params.yaml의 gate)으로 모델 버전을 판정하고 별칭을 붙인다 (05-4).

실행:
    uv run python src/promote.py <version>             # 판정만 하고, 통과하면 challenger 별칭
    uv run python src/promote.py <version> --approve   # 사람이 승인 → champion으로 승격

기준을 하나라도 통과하지 못하면 종료 코드 1로 멈춘다.
"""

import getpass
import sys
from datetime import datetime

import mlflow
import yaml

NAME = "flowers-classifier"
gate = yaml.safe_load(open("params.yaml"))["gate"]
version = sys.argv[1]
approve = "--approve" in sys.argv

client = mlflow.MlflowClient()
tags = client.get_model_version(NAME, version).tags
if "test_f1_macro" not in tags:
    sys.exit(f"v{version}는 아직 평가되지 않았습니다. 먼저 src/evaluate.py {version}를 실행하세요.")

try:
    champion = client.get_model_version_by_alias(NAME, "champion")
except mlflow.exceptions.MlflowException:
    champion = None

f1, latency = float(tags["test_f1_macro"]), float(tags["test_latency_ms"])
checks = [
    (f"테스트 F1 {f1:.4f} >= {gate['min_test_f1']}", f1 >= gate["min_test_f1"]),
    (f"CPU 추론 {latency:.1f}ms <= {gate['max_latency_ms']}ms", latency <= gate["max_latency_ms"]),
]
if gate["require_clean_git"]:
    checks.append((f"커밋된 코드로 학습 (git_dirty={tags['git_dirty']})", tags["git_dirty"] == "False"))
if champion and champion.version != version:
    champ_f1 = float(champion.tags["test_f1_macro"])
    checks.append((f"현재 champion v{champion.version}(F1 {champ_f1:.4f}) 대비 하락 <= {gate['max_f1_drop']}",
                   champ_f1 - f1 <= gate["max_f1_drop"]))

print(f"{NAME} v{version} 배포 승인 판정")
for text, ok in checks:
    print(f"  [{'OK' if ok else '불가'}] {text}")
if not all(ok for _, ok in checks):
    sys.exit("판정: 불가 — 별칭을 바꾸지 않습니다.")

if not approve:
    client.set_registered_model_alias(NAME, "challenger", version)
    print("판정: 통과 — challenger 별칭을 붙였습니다. 승격하려면 --approve를 붙여 다시 실행하세요.")
    sys.exit(0)

# 사람의 승인: 이전 champion을 previous로 남기고 새 버전을 champion으로
if champion and champion.version != version:
    client.set_registered_model_alias(NAME, "previous", champion.version)
client.set_registered_model_alias(NAME, "champion", version)
if "challenger" in client.get_registered_model(NAME).aliases:
    client.delete_registered_model_alias(NAME, "challenger")
client.set_model_version_tag(NAME, version, "approved_by", getpass.getuser())
client.set_model_version_tag(NAME, version, "approved_at", datetime.now().isoformat(timespec="seconds"))
print(f"판정: 통과 — v{version}를 champion으로 승격했습니다"
      + (f" (이전 champion v{champion.version} → previous)" if champion and champion.version != version else ""))
