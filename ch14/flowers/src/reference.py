"""드리프트 검사의 기준 데이터를 모델 버전마다 만든다 (12-3).

기준 = 그 모델이 학습 때 본 분포. 검증 세트 사진의 입력 특성과, 그 모델이 검증 세트에 낸 예측(클래스·확신도)이다.
학습 세트가 아니라 검증 세트를 쓰는 이유: 학습 세트에서는 모델이 지나치게 확신하므로 운영 확신도와 비교할 수 없다.
기준은 모델에 붙어 다닌다 — 새 모델이 champion이 되면 드리프트 검사의 기준도 함께 바뀐다.

실행 (지금 champion처럼 이 파일보다 먼저 만든 모델에 기준을 붙일 때):
    uv run python -m src.reference 7
"""

import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from PIL import Image

from src.features import image_features
from src.training import MEAN, STD

NAME = "flowers-classifier"
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def predict_folder(version: str, folder: str = "data/prepared/val") -> pd.DataFrame:
    """폴더의 사진마다 입력 특성과 모델 예측을 계산한다. 추론 서버와 같은 순서(0~1 → 정규화)로."""
    os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
    model = mlflow.pyfunc.load_model(f"models:/{NAME}/{version}")
    rows, batch = [], []
    paths = sorted(p for c in CLASSES for p in (Path(folder) / c).glob("*.jpg"))
    for path in paths:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        rows.append({"path": path.relative_to(folder).as_posix(), "label": path.parent.name, **image_features(rgb)})
        batch.append(((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))
    logits = np.concatenate([np.asarray(model.predict(np.stack(batch[i:i + 64]))) for i in range(0, len(batch), 64)])
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    df = pd.DataFrame(rows)
    df["predicted_class"] = [CLASSES[i] for i in probs.argmax(1)]
    df["confidence"] = probs.max(1).round(4)
    return df


def build(version: str) -> str:
    """기준 데이터를 MLflow에 남기고, 모델 버전 태그(reference_run_id)로 연결한다."""
    df = predict_folder(version)
    mlflow.set_experiment("flowers-reference")
    with mlflow.start_run(run_name=f"{NAME}-v{version}") as run:
        mlflow.log_text(df.to_csv(index=False), "reference.csv")
        mlflow.log_metrics({"rows": len(df), "brightness_mean": df["brightness"].mean(),
                            "confidence_mean": df["confidence"].mean(),
                            "low_confidence_rate": (df["confidence"] < 0.6).mean()})
    mlflow.MlflowClient().set_model_version_tag(NAME, version, "reference_run_id", run.info.run_id)
    print(f"v{version} 기준 데이터 {len(df)}행 → run {run.info.run_id[:8]}")
    return run.info.run_id


if __name__ == "__main__":
    build(sys.argv[1])
