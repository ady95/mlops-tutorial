"""번호판 인식 API (14-2). 모델 두 개를 이어서 서빙한다: 검출 → 자르기 → 인식.

실행:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8010

두 모델은 레지스트리에서 각자의 별칭(champion)으로 불러오고, /reload?model=recognizer 처럼 하나씩 바꾼다.
자르기·전처리는 학습 코드와 같은 src/models.py를 쓴다 — 두 모델 사이의 약속(계약)이 한 곳에만 있게.
"""

import io
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics import FORMAT_OK, MODEL_VERSION, REQUESTS, STAGE_SECONDS
from src.models import crop, decode, det_input, rec_input

NAMES = {"detector": os.environ.get("DETECTOR_NAME", "plate-detector"),
         "recognizer": os.environ.get("RECOGNIZER_NAME", "plate-recognizer")}
ALIAS = os.environ.get("MODEL_ALIAS", "champion")
RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "")
PLATE = re.compile(r"^\d{2,3}[가-힣] \d{4}$")       # 정답 없이도 볼 수 있는 품질 신호: 번호판 형식에 맞는가
WARMUP = {"detector": np.zeros((1, 3, 224, 224), np.float32), "recognizer": np.zeros((1, 1, 32, 128), np.float32)}
os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", torch.get_num_threads())))

state = {}
lock = threading.Lock()


def load(kind: str) -> str:
    """별칭이 지금 가리키는 버전을 찾아 그 번호로 불러온다. 다른 모델은 건드리지 않는다."""
    mv = mlflow.MlflowClient().get_model_version_by_alias(NAMES[kind], ALIAS)
    model = mlflow.pyfunc.load_model(f"models:/{NAMES[kind]}/{mv.version}")
    model.predict(WARMUP[kind])
    with lock:
        state[kind] = {"model": model, "version": mv.version, "loaded_at": datetime.now().isoformat(timespec="seconds")}
    MODEL_VERSION.labels(kind).set(int(mv.version))
    return mv.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    load("detector")
    load("recognizer")
    yield


app = FastAPI(title="plate API", lifespan=lifespan)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"} | {k: v["version"] for k, v in state.items()}


@app.get("/model")
def model_info():
    return {k: {"name": NAMES[k], "alias": ALIAS, "version": v["version"], "loaded_at": v["loaded_at"]}
            for k, v in state.items()}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(file.file.read(5 * 1024 * 1024))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        REQUESTS.labels("400").inc()
        raise HTTPException(status_code=400, detail="이미지 파일로 읽을 수 없습니다")
    with lock:                                     # 한 요청은 처음부터 끝까지 같은 두 버전으로
        det, rec = state["detector"], state["recognizer"]

    t = time.perf_counter()
    box = np.asarray(det["model"].predict(det_input(img)[None]))[0].tolist()
    STAGE_SECONDS.labels("detector").observe(time.perf_counter() - t)
    t = time.perf_counter()
    text = decode(np.asarray(rec["model"].predict(rec_input(crop(img, box))[None]))[0])
    STAGE_SECONDS.labels("recognizer").observe(time.perf_counter() - t)

    ok = bool(PLATE.match(text))
    FORMAT_OK.labels(str(ok).lower(), det["version"], rec["version"]).inc()
    REQUESTS.labels("200").inc()
    return {"text": text, "format_ok": ok, "bbox": [round(v, 4) for v in box],
            "versions": {"detector": det["version"], "recognizer": rec["version"]},
            "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


@app.post("/reload")
def reload(model: str, x_reload_token: str = Header(default="")):
    """한 모델만 다시 불러온다: /reload?model=detector 또는 /reload?model=recognizer."""
    if not RELOAD_TOKEN or x_reload_token != RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="reload 권한이 없습니다")
    if model not in NAMES:
        raise HTTPException(status_code=400, detail="model은 detector 또는 recognizer")
    before = state[model]["version"]
    return {"model": model, "before": before, "after": load(model)}
