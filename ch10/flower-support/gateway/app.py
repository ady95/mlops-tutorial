"""고객 문의 JSON 추출 게이트웨이: vLLM 앞에서 프롬프트와 어댑터를 고르는 얇은 서버 (10-5).

클라이언트는 문의만 보낸다. 어떤 어댑터와 어떤 프롬프트를 쓸지는 레지스트리의 별칭이 정하고,
/reload를 부르면 champion 어댑터와 production 프롬프트로 바꾼다. 6장 추론 서버의 /reload와 같은 역할이다.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import mlflow
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from adapter_for_vllm import convert

VLLM = os.environ.get("VLLM_URL", "http://localhost:8001")
BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "qwen3.5-4b")
MODEL = os.environ.get("MODEL_NAME", "flower-support-adapter")
PROMPT = os.environ.get("PROMPT_NAME", "flower-support-extract")
RELEASES = Path(os.environ.get("RELEASES_DIR", "/releases"))       # vLLM 컨테이너도 같은 경로로 본다
TOKEN = os.environ.get("RELOAD_TOKEN", "")
CANARY = "주문번호 FL-000123 장미 꽃다발이 아직 안 왔어요. 내일 행사라 오늘 꼭 확인해 주세요."
FIELDS = ["category", "urgency", "sentiment", "order_id", "product", "summary"]

client = mlflow.MlflowClient()
current: dict = {}            # {"model_version", "prompt_version", "lora_name", "template"}


def chat(model: str, messages: list[dict]) -> str:
    r = httpx.post(f"{VLLM}/v1/chat/completions", timeout=60, json={
        "model": model, "messages": messages, "temperature": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def loaded_models() -> set[str]:
    return {m["id"] for m in httpx.get(f"{VLLM}/v1/models", timeout=10).json()["data"]}


def prepare(version: str) -> str:
    """레지스트리의 어댑터를 받아 vLLM용 이름으로 바꾸고 vLLM에 올린다. 이미 올라가 있으면 그대로 쓴다."""
    lora_name = f"{MODEL}-v{version}"
    if lora_name in loaded_models():
        return lora_name
    src = Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL}/{version}", dst_path=str(RELEASES / f"v{version}")))
    convert(src / "peft", RELEASES / f"v{version}" / "adapter-vllm")
    r = httpx.post(f"{VLLM}/v1/load_lora_adapter", timeout=120,
                   json={"lora_name": lora_name, "lora_path": str(RELEASES / f"v{version}" / "adapter-vllm")})
    r.raise_for_status()
    return lora_name


def activate(model_version: str, prompt_version: str) -> dict:
    lora_name = prepare(model_version)
    template = mlflow.genai.load_prompt(f"prompts:/{PROMPT}/{prompt_version}")
    messages = template.format(inquiry=CANARY)
    # 적용 확인: 어댑터가 실제로 출력에 영향을 주는지 (10-1에서 이름이 안 맞아 조용히 무시된 사례)
    tuned, base = chat(lora_name, messages), chat(BASE_MODEL, messages)
    if tuned == base:
        raise HTTPException(500, f"{lora_name}의 출력이 베이스 모델과 같습니다. 어댑터가 적용되지 않았습니다.")
    try:
        parsed = json.loads(tuned)
        assert set(parsed) == set(FIELDS)
    except (json.JSONDecodeError, AssertionError):
        raise HTTPException(500, f"{lora_name}의 출력이 형식에 맞지 않습니다: {tuned[:200]}")
    return {"model_version": model_version, "prompt_version": prompt_version, "lora_name": lora_name,
            "template": template}


def resolve() -> tuple[str, str]:
    mv = client.get_model_version_by_alias(MODEL, "champion")
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT}@production")
    return mv.version, str(prompt.version)


@asynccontextmanager
async def lifespan(_):
    current.update(activate(*resolve()))      # 시작할 때 champion + production으로
    yield


app = FastAPI(title="flower-support gateway", lifespan=lifespan)


class Inquiry(BaseModel):
    inquiry: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": current.get("model_version"),
            "prompt_version": current.get("prompt_version")}


@app.post("/extract")
def extract(req: Inquiry):
    snapshot = dict(current)            # 요청 도중 /reload가 일어나도 한 요청은 한 버전으로 끝난다
    start = time.perf_counter()
    messages = snapshot["template"].format(inquiry=req.inquiry)
    try:
        text = chat(snapshot["lora_name"], messages)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise
        # vLLM이 다시 시작되면 실행 중에 올린 어댑터가 사라진다 → 같은 버전을 다시 올리고 한 번 더
        prepare(snapshot["model_version"])
        text = chat(snapshot["lora_name"], messages)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(502, "모델 출력이 JSON이 아닙니다")
    return {"result": result, "model_version": snapshot["model_version"],
            "prompt_version": snapshot["prompt_version"], "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


@app.post("/reload")
def reload(x_reload_token: str = Header(default="")):
    if not TOKEN or x_reload_token != TOKEN:
        raise HTTPException(403, "reload 토큰이 맞지 않습니다")
    before = {k: current.get(k) for k in ("model_version", "prompt_version")}
    start = time.perf_counter()
    new = activate(*resolve())          # 준비와 확인이 끝난 뒤에야 바꾼다 — 실패하면 지금 버전 그대로
    current.update(new)
    return {"before": before, "after": {k: new[k] for k in ("model_version", "prompt_version")},
            "seconds": round(time.perf_counter() - start, 2)}
