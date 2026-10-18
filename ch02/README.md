# 2장. 로컬 MLOps 개발환경 구축

| 폴더 | 내용 | 관련 절 |
|---|---|---|
| `mlflow-server/` | MLflow 추적 서버 (PostgreSQL + RustFS) Docker Compose | 02-4 |
| `first-run/` | Fashion-MNIST 분류 모델 학습 및 MLflow 기록 | 02-5 |

## 빠른 실행

```bash
# 1) MLflow 서버
cd mlflow-server
cp .env.example .env
docker compose up -d          # http://localhost:5000

# 2) 첫 학습 (CPU)
cd ../first-run
uv sync --extra cpu           # NVIDIA GPU: uv sync --extra cu130
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run train.py --epochs 3
```

## 기준 버전

MLflow 3.17.1 · PyTorch 2.14.1 · Python 3.12 · uv 0.13.1 · PostgreSQL 17 · RustFS 1.0.3
