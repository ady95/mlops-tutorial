"""MLflow에 기록된 파인튜닝 실험을 한 표로 비교한다 (08-5).

실행:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run python -m src.compare
"""

import mlflow

runs = mlflow.search_runs(experiment_names=["ch08-flower-support"],
                          filter_string="tags.stage != 'evaluation' AND attributes.status = 'FINISHED'",
                          order_by=["metrics.test_all_fields DESC"])
columns = {
    "tags.mlflow.runName": "실행",
    "tags.gpu": "GPU",
    "metrics.train_minutes": "학습(분)",
    "metrics.gpu_peak_allocated_gb": "메모리(GB)",
    "params.trainable_params": "학습 파라미터",
    "metrics.adapter_mb": "어댑터(MB)",
    "metrics.eval_loss": "검증 손실",
    "metrics.test_urgency": "긴급도",
    "metrics.test_all_fields": "전체 정답",
}
table = runs[list(columns)].rename(columns=columns)
table["GPU"] = table["GPU"].fillna("").str.replace("NVIDIA GeForce ", "")
print(table.round(4).to_string(index=False))
