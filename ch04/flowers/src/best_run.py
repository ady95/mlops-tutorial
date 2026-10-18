"""MLflow 검색 API로 조건에 맞는 실행을 찾고 비교한다 (04-4).

실행:
    uv run python src/best_run.py                                  # 전체 비교
    uv run python src/best_run.py "metrics.cpu_latency_ms < 20"    # 조건 걸기
"""

import sys

import mlflow

filter_string = sys.argv[1] if len(sys.argv) > 1 else ""
runs = mlflow.search_runs(
    experiment_names=["ch04-flowers"],
    filter_string=filter_string,
    order_by=["metrics.val_f1_macro DESC"],
)
runs = runs[runs["params.epochs"] == "5"]          # 5에폭 실행만 비교

cols = {
    "params.model": "model",
    "params.freeze_backbone": "freeze",
    "params.lr": "lr",
    "tags.gpu": "gpu",
    "metrics.val_accuracy": "val_acc",
    "metrics.val_f1_macro": "val_f1",
    "metrics.train_seconds": "train_s",
    "metrics.cpu_latency_ms": "latency_ms",
}
table = runs[list(cols)].rename(columns=cols)
table["gpu"] = table["gpu"].str.replace("NVIDIA GeForce ", "")
table = table.round({"val_acc": 4, "val_f1": 4, "train_s": 0, "latency_ms": 1})
print(table.to_string(index=False))

if len(runs):
    best = runs.iloc[0]
    print(f"\nbest: {best['params.model']} (freeze={best['params.freeze_backbone']}) "
          f"val_f1={best['metrics.val_f1_macro']:.4f} run_id={best['run_id']}")
