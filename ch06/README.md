# 6장. 모델 배포와 API 서비스

5장의 레지스트리에서 `champion` 모델을 불러와 FastAPI 추론 서버로 서비스하고, Docker 이미지로 묶고, ONNX로 최적화한 뒤 부하 테스트를 합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flowers/app/main.py` | FastAPI 추론 서버 (`/health`, `/model`, `/predict`, `/reload`) | 06-1 |
| `flowers/app/preprocess.py` | 학습 때와 같은 전처리 (224 크롭, ImageNet 정규화) | 06-1 |
| `flowers/Dockerfile`, `compose.yaml` | 추론 서버 이미지와 실행 구성 | 06-2 |
| `flowers/src/export_onnx.py` | champion → ONNX 변환, PyTorch와 비교, 레지스트리 등록 | 06-4 |
| `flowers/src/api_check.py` | 테스트 세트 원본 사진으로 API 정확도 확인 | 06-1, 06-5 |
| `flowers/src/loadtest.py` | 동시 요청 부하 테스트 (처리량, p50/p95/p99) | 06-5 |
| `flowers/src/download.py` 등 | 테스트용 사진 준비 (3장 파이프라인) | 3장 |

## 빠른 실행

```bash
# 2장 MLflow 서버와 5장의 champion 모델이 있어야 합니다
cd flowers
uv sync --extra cpu
export MLFLOW_TRACKING_URI=http://localhost:5000

# 직접 실행
RELOAD_TOKEN=change-me uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Docker로 실행
echo "RELOAD_TOKEN=change-me" > .env
docker compose up -d --build
curl -F "file=@some_flower.jpg" http://localhost:8000/predict

# ONNX 변환 후 ONNX 모델로 서빙 (서버 코드는 그대로)
uv run python -m src.export_onnx
MODEL_NAME=flowers-classifier-onnx docker compose up -d
```
