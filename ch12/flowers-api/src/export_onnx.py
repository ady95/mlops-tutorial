"""champion 모델을 ONNX로 변환하고, PyTorch와 결과를 비교한 뒤 레지스트리에 등록한다 (06-4).

실행 (app 패키지를 찾을 수 있도록 프로젝트 루트에서 모듈로 실행):
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run python -m src.export_onnx
"""

import os
import time

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import datasets, transforms

from app.preprocess import MEAN, STD

NAME, ONNX_NAME = "flowers-classifier", "flowers-classifier-onnx"
os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")

client = mlflow.MlflowClient()
src = client.get_model_version_by_alias(NAME, "champion")
# pt2로 저장된 모델은 torch.export로 그래프가 고정되어 있다. 학습 코드가 추론 모드로 저장했으므로
# 변환 때 나오는 "training mode" 경고는 무시해도 되며, eval()은 호출할 수 없다(NotImplementedError)
model = mlflow.pytorch.load_model(f"models:/{NAME}/{src.version}", map_location="cpu")

# 1. 변환: 배치 크기는 바뀔 수 있게(dynamic) 둔다
example = torch.randn(4, 3, 224, 224)
batch = torch.export.Dim("batch", min=1, max=64)
torch.onnx.export(model, (example,), "flowers.onnx", dynamo=True,
                  input_names=["image"], output_names=["logits"],
                  dynamic_shapes=({0: batch},))
# 그래프는 flowers.onnx에, 가중치는 flowers.onnx.data에 따로 저장된다
print(f"saved flowers.onnx ({os.path.getsize('flowers.onnx') / 1e6:.1f}MB) "
      f"+ flowers.onnx.data ({os.path.getsize('flowers.onnx.data') / 1e6:.1f}MB)")

# 2. 테스트 세트에서 PyTorch와 비교
test_set = datasets.ImageFolder("data/prepared/test", transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist())]))
x = np.stack([img.numpy() for img, _ in test_set])
labels = np.array(test_set.targets)

session = ort.InferenceSession("flowers.onnx", providers=["CPUExecutionProvider"])
with torch.no_grad():
    pt_logits = np.concatenate([model(torch.from_numpy(x[i:i + 64])).numpy() for i in range(0, len(x), 64)])
onnx_logits = np.concatenate([session.run(None, {"image": x[i:i + 64]})[0] for i in range(0, len(x), 64)])


def latency_ms(fn, runs=50):
    for _ in range(5):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


one = x[:1]
with torch.no_grad():
    pt_ms = latency_ms(lambda: model(torch.from_numpy(one)))
onnx_ms = latency_ms(lambda: session.run(None, {"image": one}))

metrics = {
    "max_abs_diff": float(np.abs(pt_logits - onnx_logits).max()),
    "prediction_agreement": float((pt_logits.argmax(1) == onnx_logits.argmax(1)).mean()),
    "pytorch_accuracy": float((pt_logits.argmax(1) == labels).mean()),
    "onnx_accuracy": float((onnx_logits.argmax(1) == labels).mean()),
    "pytorch_latency_ms": pt_ms,
    "onnx_latency_ms": onnx_ms,
}
for k, v in metrics.items():
    print(f"{k:<22} {v:.6f}")

# 3. ONNX 모델을 기록하고 별도 이름으로 등록 (원본 버전을 태그로 남긴다)
mlflow.set_experiment("ch06-onnx")
with mlflow.start_run(run_name=f"onnx-from-v{src.version}"):
    mlflow.log_metrics(metrics)
    # log_model은 파일 경로가 아니라 모델 객체를 받는다. onnx.load가 가중치 파일까지 읽어 온다
    info = mlflow.onnx.log_model(onnx.load("flowers.onnx"), name="model", input_example=x[:4])
mv = mlflow.register_model(info.model_uri, ONNX_NAME,
                           tags={"source_model": NAME, "source_version": src.version})
print(f"registered {ONNX_NAME} version {mv.version} (from {NAME} v{src.version})")

# 원본(champion)은 이미 5장의 승인 기준을 통과했다. 변환본은 "원본과 결과가 같은가"만 확인하면 된다
if metrics["prediction_agreement"] == 1.0 and metrics["onnx_accuracy"] == metrics["pytorch_accuracy"]:
    client.set_registered_model_alias(ONNX_NAME, "champion", mv.version)
    print(f"{ONNX_NAME} v{mv.version} → champion (원본과 예측 100% 일치)")
else:
    print("원본과 예측이 다릅니다. champion 별칭을 붙이지 않습니다.")
