"""학습 실행의 모델을 MLflow 모델 레지스트리에 새 버전으로 등록한다 (05-2).

실행:
    uv run python src/register.py <run_id>
"""

import sys

import mlflow

NAME = "flowers-classifier"

run = mlflow.get_run(sys.argv[1])
p, t, m = run.data.params, run.data.tags, run.data.metrics

# 버전 태그: 이 버전이 어떤 코드·데이터·설정으로 만들어졌고 검증 성능이 어땠는지
tags = {
    "model": p["model"], "lr": p["lr"], "freeze_backbone": p["freeze_backbone"],
    "git_commit": t["git_commit"], "git_dirty": t["git_dirty"],
    "data_prepared_md5": t["data_prepared_md5"],
    "val_f1_macro": f"{m['val_f1_macro']:.4f}",
}
# MLflow 3에서는 모델이 실행의 아티팩트가 아니라 별도의 '기록된 모델(m-...)'로 저장된다
model_id = run.outputs.model_outputs[0].model_id
mv = mlflow.register_model(f"models:/{model_id}", NAME, tags=tags)

desc = (f"{p['model']} {'분류층만' if p['freeze_backbone'] == 'True' else '전체 미세조정'} "
        f"lr={p['lr']}, 검증 F1 {m['val_f1_macro']:.4f} (run {run.info.run_id[:8]})")
mlflow.MlflowClient().update_model_version(NAME, mv.version, description=desc)
print(f"registered {NAME} version {mv.version}: {desc}")
