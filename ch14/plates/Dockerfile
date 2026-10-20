# 번호판 인식 API 이미지 (14-2) — 6장 flowers-api와 같은 구성
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cpu

# 서버 코드 + 학습과 같은 전처리(src/models.py, src/synth.py) — 두 모델 사이의 계약을 한 곳에서
COPY app ./app
COPY src/__init__.py src/models.py src/synth.py ./src/

ENV PATH="/app/.venv/bin:$PATH" \
    MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu
EXPOSE 8010
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
