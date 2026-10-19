# 게이트웨이 이미지 (10-5). 빌드는 flower-support 폴더에서: docker compose -f serving/compose.yaml build
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
WORKDIR /app
COPY gateway/pyproject.toml gateway/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY gateway/app.py src/adapter_for_vllm.py ./
CMD ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
