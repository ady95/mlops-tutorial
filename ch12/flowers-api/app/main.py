"""꽃 분류 추론 API 서버 (06-1) + Prometheus 지표 (12-1) + 예측 기록 (12-3).

실행:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from PIL import UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.features import image_features
from app.metrics import BRIGHTNESS, CONFIDENCE, LATENCY, MODEL_VERSION, PREDICTIONS, REQUESTS
from app.preprocess import CLASSES, MEAN, STD, softmax, to_tensor

MODEL_NAME = os.environ.get("MODEL_NAME", "flowers-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
# 비어 있으면 /reload를 막는다
RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "")
MAX_BYTES = 10 * 1024 * 1024
os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", torch.get_num_threads())))

PREDICTION_LOG_DIR = os.environ.get("PREDICTION_LOG_DIR", "")   # 비어 있으면 예측 기록을 남기지 않는다 (12-3)

state = {}
lock = threading.Lock()
log_lock = threading.Lock()


def to_logits(output) -> np.ndarray:
    """pyfunc 출력 형식은 모델 형식마다 다르다: PyTorch는 배열, ONNX는 {출력이름: 배열}."""
    if isinstance(output, dict):
        output = next(iter(output.values()))
    return np.asarray(output)


def load_model():
    """별칭이 지금 가리키는 버전을 찾아, 그 버전 번호로 고정해서 불러온다."""
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    # 첫 요청이 느리지 않게 미리 한 번
    model.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    with lock:
        state.update(model=model, version=mv.version, description=mv.description,
                     loaded_at=datetime.now().isoformat(timespec="seconds"))
    MODEL_VERSION.set(int(mv.version))
    return mv.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="flowers-classifier API", lifespan=lifespan)


def log_prediction(record: dict):
    """예측 기록을 날짜별 JSONL 파일에 한 줄씩 더한다 (12-3 드리프트 검사의 입력). 사진은 남기지 않는다."""
    if not PREDICTION_LOG_DIR:
        return
    now = datetime.now(timezone.utc)
    line = json.dumps({"ts": now.isoformat(timespec="seconds"), **record}, ensure_ascii=False)
    with log_lock, open(Path(PREDICTION_LOG_DIR) / f"predictions-{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")


@app.get("/metrics")
async def metrics():
    """Prometheus가 가져갈 지표 (12-1). app.mount로 붙이면 /metrics/로 넘겨주는 307 응답이 끼어든다.
    async로 둔다: 예측(동기 함수)이 작업 스레드를 모두 차지해도 지표는 바로 답한다 (12-2)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["version"]}


@app.get("/model")
def model_info():
    return {"name": MODEL_NAME, "alias": MODEL_ALIAS, "version": state["version"],
            "description": state["description"], "loaded_at": state["loaded_at"],
            "torch_threads": torch.get_num_threads()}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        REQUESTS.labels("predict", "413").inc()
        raise HTTPException(status_code=413, detail="이미지는 10MB 이하만 받습니다")
    try:
        x = to_tensor(data)
    except (UnidentifiedImageError, OSError):
        REQUESTS.labels("predict", "400").inc()
        raise HTTPException(status_code=400, detail="이미지 파일로 읽을 수 없습니다")
    features = image_features(x[0].transpose(1, 2, 0) * STD + MEAN)          # 정규화를 되돌려 0~1 범위로
    BRIGHTNESS.observe(features["brightness"])

    # /reload 중에도 일관된 모델·버전으로 응답
    with lock:
        model, version = state["model"], state["version"]
    probs = softmax(to_logits(model.predict(x)))[0]
    top = int(probs.argmax())
    REQUESTS.labels("predict", "200").inc()
    PREDICTIONS.labels(CLASSES[top], version).inc()
    CONFIDENCE.observe(float(probs[top]))
    LATENCY.observe(time.perf_counter() - start)
    log_prediction({**features, "predicted_class": CLASSES[top], "confidence": round(float(probs[top]), 4),
                    "model_version": version})
    return {"class": CLASSES[top], "probability": round(float(probs[top]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
            "model_version": version, "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


@app.post("/reload")
def reload(x_reload_token: str = Header(default="")):
    """레지스트리의 별칭이 바뀐 뒤 서버를 재시작하지 않고 새 버전을 불러온다 (5장 롤백 반영)."""
    if not RELOAD_TOKEN or x_reload_token != RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="reload 권한이 없습니다")
    before = state["version"]
    after = load_model()
    return {"before": before, "after": after}
