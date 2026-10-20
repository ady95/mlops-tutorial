# 꽃 분류 추론 서버 이미지 (06-2)
FROM python:3.12-slim

# uv 실행 파일만 공식 이미지에서 복사
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# 1) 의존성만 먼저 설치: 코드가 바뀌어도 이 층은 캐시를 재사용한다
#    uv 다운로드 캐시는 빌드 중에만 쓰는 캐시 마운트에 두어 이미지에 남기지 않는다
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cpu

# 2) 서버 코드
COPY app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
