"""MLOps 따라하기 02-5: MLflow에 기록된 모델을 불러와 예측한다.

실행:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run load_model.py <run_id>
"""

import os
import sys

import mlflow
import numpy as np
from torchvision import datasets, transforms

# CPU에서 저장한 pt2 모델은 GPU가 있는 PC에서도 CPU로 불러온다
os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

run_id = sys.argv[1]
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
images = np.stack([test_set[i][0].numpy() for i in range(8)])
labels = [test_set[i][1] for i in range(8)]

logits = np.asarray(model.predict(images))
for pred, label in zip(logits.argmax(axis=1), labels):
    mark = "OK" if pred == label else "틀림"
    print(f"정답 {CLASSES[label]:<11} 예측 {CLASSES[pred]:<11} {mark}")
