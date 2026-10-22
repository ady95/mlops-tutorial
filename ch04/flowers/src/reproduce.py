"""MLflow에 기록된 실행을 같은 코드·데이터·설정으로 다시 학습한다 (04-5).

실행:
    uv run python src/reproduce.py <run_id>
"""

import re
import subprocess
import sys

import mlflow
from train import code_and_data_version

run = mlflow.get_run(sys.argv[1])
params, tags = run.data.params, run.data.tags

# 1. 코드와 데이터가 기록 당시와 같은지 확인한다
now = code_and_data_version()
for key in ("git_commit", "data_prepared_md5"):
    same = tags.get(key) == now[key]
    print(f"{key:<18} 기록 {tags.get(key, '-')[:12]}  현재 {now[key][:12]}  {'같음' if same else '다름'}", flush=True)
    if not same:
        sys.exit(f"{key}가 다릅니다. git checkout {tags.get('git_commit', '')[:12]} && dvc checkout 후 다시 실행하세요.")

# 2. 기록된 파라미터로 같은 명령을 만든다
cmd = [sys.executable, "src/train.py", "--model", params["model"], "--epochs", params["epochs"],
       "--batch-size", params["batch_size"], "--lr", params["lr"], "--seed", params["seed"],
       "--num-workers", params["num_workers"]]
for flag in ("freeze_backbone", "deterministic"):
    if params.get(flag) == "True":
        cmd.append("--" + flag.replace("_", "-"))
print("실행:", " ".join(cmd[1:]), flush=True)
out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
print(out)

# 3. 새 실행과 원래 실행의 지표를 비교한다
# "가장 최근 실행"으로 찾으면 같은 서버에 동시에 기록하는 다른 사람의 실행이 잡힐
# 수 있어,
#    학습 스크립트가 출력한 run_id로 정확히 찾는다
new_id = re.search(r"run_id=(\w+)", out).group(1)
mlflow.MlflowClient().set_tag(new_id, "reproduced_from", run.info.run_id)
new = mlflow.get_run(new_id)
for m in ("val_accuracy", "val_f1_macro", "val_loss"):
    old_v, new_v = run.data.metrics[m], new.data.metrics[m]
    print(f"{m:<14} 원래 {old_v:.6f}  재현 {new_v:.6f}  차이 {new_v - old_v:+.6f}")
