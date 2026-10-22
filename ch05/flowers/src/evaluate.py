"""레지스트리의 모델 버전을 테스트 세트로 평가하고 결과를 버전 태그로 남긴다 (05-3).

실행:
    uv run python src/evaluate.py <version>

테스트 세트(data/prepared/test)는 학습·모델 선택에 한 번도 쓰지 않은 데이터다.
"""

import os
import sys
import time

import mlflow
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from train import MEAN, STD, code_and_data_version

NAME = "flowers-classifier"
# 서빙과 같은 CPU 기준
os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")

version = sys.argv[1]
client = mlflow.MlflowClient()
mv = client.get_model_version(NAME, version)
model = mlflow.pyfunc.load_model(f"models:/{NAME}/{version}")

test_set = datasets.ImageFolder("data/prepared/test",
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize(MEAN, STD)]))
images = np.stack([x.numpy() for x, _ in test_set])
labels = np.array(test_set.targets)

preds = np.concatenate([np.asarray(model.predict(images[i:i + 64])).argmax(1)
                        for i in range(0, len(images), 64)])

# 사진 1장 추론 시간 (워밍업 5회 후 50회 중앙값)
one = images[:1]
for _ in range(5):
    model.predict(one)
times = []
for _ in range(50):
    start = time.perf_counter()
    model.predict(one)
    times.append((time.perf_counter() - start) * 1000)
latency = float(np.median(times))

mlflow.set_experiment("ch05-evaluation")
with mlflow.start_run(run_name=f"{NAME}-v{version}") as run:
    mlflow.log_params({"model_name": NAME, "model_version": version, "test_size": len(labels)})
    mlflow.set_tags({"eval_data_prepared_md5": code_and_data_version()["data_prepared_md5"]})
    # 예측값을 직접 넘기면 MLflow 표준 분류 평가(정확도·F1·혼동 행렬 등)를 그대로
    # 쓸 수 있다.
    # 다중 분류의 기본 평균 방식은 weighted라서, 4장의 검증 F1과 같은 macro로
    # 지정한다.
    result = mlflow.models.evaluate(
        data=pd.DataFrame({"label": labels, "prediction": preds}),
        targets="label", predictions="prediction", model_type="classifier",
        evaluator_config={"average": "macro", "label_list": list(range(len(test_set.classes)))},
    )
    mlflow.log_metric("latency_ms", latency)

metrics = {"test_accuracy": result.metrics["accuracy_score"],
           "test_f1_macro": result.metrics["f1_score"], "test_latency_ms": latency}
for k, v in metrics.items():
    client.set_model_version_tag(NAME, version, k, f"{v:.4f}")
client.set_model_version_tag(NAME, version, "eval_run_id", run.info.run_id)
print(f"{NAME} v{version}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
