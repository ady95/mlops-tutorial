"""레지스트리 등록·평가·승인 판정 — 5장 스크립트를 함수로 정리했다 (07-2).

judge()는 MLflow에 접속하지 않는 순수 함수라서 CI에서 단위 테스트할 수 있다 (07-4).
"""

import os
import time

import mlflow
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from src.training import MEAN, STD

NAME = "flowers-classifier"


def register(run_id: str) -> str:
    run = mlflow.get_run(run_id)
    p, t, m = run.data.params, run.data.tags, run.data.metrics
    tags = {"model": p["model"], "lr": p["lr"], "freeze_backbone": p["freeze_backbone"],
            "git_commit": t["git_commit"], "git_dirty": t["git_dirty"],
            "data_prepared_md5": t["data_prepared_md5"], "val_f1_macro": f"{m['val_f1_macro']:.4f}"}
    for key in ("zenml_pipeline_run_id",):
        if key in t:
            tags[key] = t[key]
    model_id = run.outputs.model_outputs[0].model_id
    mv = mlflow.register_model(f"models:/{model_id}", NAME, tags=tags)
    mlflow.MlflowClient().update_model_version(
        NAME, mv.version, description=f"{p['model']} lr={p['lr']}, 검증 F1 {m['val_f1_macro']:.4f} (run {run_id[:8]})")
    return mv.version


def evaluate(version: str) -> dict:
    os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
    client = mlflow.MlflowClient()
    model = mlflow.pyfunc.load_model(f"models:/{NAME}/{version}")
    test_set = datasets.ImageFolder("data/prepared/test", transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    images = np.stack([x.numpy() for x, _ in test_set])
    labels = np.array(test_set.targets)
    preds = np.concatenate([np.asarray(model.predict(images[i:i + 64])).argmax(1)
                            for i in range(0, len(images), 64)])
    for _ in range(5):
        model.predict(images[:1])
    times = []
    for _ in range(50):
        start = time.perf_counter()
        model.predict(images[:1])
        times.append((time.perf_counter() - start) * 1000)

    mlflow.set_experiment("ch07-evaluation")
    with mlflow.start_run(run_name=f"{NAME}-v{version}"):
        result = mlflow.models.evaluate(
            data=pd.DataFrame({"label": labels, "prediction": preds}), targets="label",
            predictions="prediction", model_type="classifier",
            evaluator_config={"average": "macro", "label_list": list(range(len(test_set.classes)))})
    metrics = {"test_accuracy": result.metrics["accuracy_score"], "test_f1_macro": result.metrics["f1_score"],
               "test_latency_ms": float(np.median(times))}
    for k, v in metrics.items():
        client.set_model_version_tag(NAME, version, k, f"{v:.4f}")
    return metrics


def judge(tags: dict, champion_tags: dict | None, gate: dict) -> list[tuple[str, bool]]:
    """배포 승인 기준 판정. (설명, 통과 여부) 목록을 돌려준다."""
    f1, latency = float(tags["test_f1_macro"]), float(tags["test_latency_ms"])
    checks = [(f"테스트 F1 {f1:.4f} >= {gate['min_test_f1']}", f1 >= gate["min_test_f1"]),
              (f"CPU 추론 {latency:.1f}ms <= {gate['max_latency_ms']}ms", latency <= gate["max_latency_ms"])]
    if gate["require_clean_git"]:
        checks.append((f"커밋된 코드로 학습 (git_dirty={tags['git_dirty']})", tags["git_dirty"] == "False"))
    if champion_tags is not None:
        champ_f1 = float(champion_tags["test_f1_macro"])
        checks.append((f"현재 champion(F1 {champ_f1:.4f}) 대비 하락 <= {gate['max_f1_drop']}",
                       champ_f1 - f1 <= gate["max_f1_drop"]))
    return checks


def champion_tags() -> tuple[str | None, dict | None]:
    try:
        mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, "champion")
        return mv.version, mv.tags
    except mlflow.exceptions.MlflowException:
        return None, None
