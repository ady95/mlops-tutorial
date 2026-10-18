"""서비스 쪽 코드의 예: 버전 번호 대신 champion 별칭으로 모델을 불러와 예측한다 (05-1, 05-5).

실행:
    uv run python src/predict_champion.py
"""

import os

import mlflow
import numpy as np
from torchvision import datasets, transforms
from train import MEAN, STD

NAME = "flowers-classifier"
os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")

mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, "champion")
model = mlflow.pyfunc.load_model(f"models:/{NAME}@champion")

test_set = datasets.ImageFolder("data/prepared/test",
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize(MEAN, STD)]))
images = np.stack([test_set[i][0].numpy() for i in range(0, len(test_set), 50)])
labels = [test_set.targets[i] for i in range(0, len(test_set), 50)]
preds = np.asarray(model.predict(images)).argmax(1)

print(f"champion = v{mv.version} ({mv.description})")
print(f"표본 {len(labels)}장 중 {int((preds == np.array(labels)).sum())}장 정답")
