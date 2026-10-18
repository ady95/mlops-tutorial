"""꽃 분류 추론 API 서버 (06-1).

실행:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.preprocess import CLASSES, softmax, to_tensor

MODEL_NAME = os.environ.get("MODEL_NAME", "flowers-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "")               # 비어 있으면 /reload를 막는다
MAX_BYTES = 10 * 1024 * 1024
os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", torch.get_num_threads())))

state = {}
lock = threading.Lock()


def to_logits(output) -> np.ndarray:
    """pyfunc 출력 형식은 모델 형식마다 다르다: PyTorch는 배열, ONNX는 {출력이름: 배열}."""
    if isinstance(output, dict):
        output = next(iter(output.values()))
    return np.asarray(output)


def load_model():
    """별칭이 지금 가리키는 버전을 찾아, 그 버전 번호로 고정해서 불러온다."""
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    model.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))   # 첫 요청이 느리지 않게 미리 한 번
    with lock:
        state.update(model=model, version=mv.version, description=mv.description,
                     loaded_at=datetime.now().isoformat(timespec="seconds"))
    return mv.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="flowers-classifier API", lifespan=lifespan)


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
        raise HTTPException(status_code=413, detail="이미지는 10MB 이하만 받습니다")
    try:
        x = to_tensor(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="이미지 파일로 읽을 수 없습니다")

    with lock:                                     # /reload 중에도 일관된 모델·버전으로 응답
        model, version = state["model"], state["version"]
    probs = softmax(to_logits(model.predict(x)))[0]
    top = int(probs.argmax())
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
