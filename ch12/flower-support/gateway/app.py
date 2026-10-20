"""고객 문의 JSON 추출 게이트웨이 + 운영 기록 (10-5, 11장) + Prometheus 지표 (12-1).

10장의 게이트웨이에 더한 것:
  - 요청마다 MLflow 트레이스를 남긴다 (모델·프롬프트 버전, 토큰 사용량, 지연, 규칙 검사, 분류 확신도)
  - 모델에 보내기 전과 기록하기 전에 민감정보를 가린다
  - 응답에 request_id(트레이스 ID)를 담아, 고객 문의 처리 결과를 나중에 찾아볼 수 있게 한다
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import mlflow
from fastapi import FastAPI, Header, HTTPException, Request, Response
from mlflow.entities import AssessmentSource, SpanType
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from adapter_for_vllm import convert
from guards import category_confidence, check_output, injection_suspected, mask_pii
from metrics import CONFIDENCE, IN_FLIGHT, LATENCY, REQUESTS, SIGNALS, TOKENS, TOTAL_LATENCY

VLLM = os.environ.get("VLLM_URL", "http://localhost:8001")
BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "qwen3.5-4b")
MODEL = os.environ.get("MODEL_NAME", "flower-support-adapter")
PROMPT = os.environ.get("PROMPT_NAME", "flower-support-extract")
RELEASES = Path(os.environ.get("RELEASES_DIR", "/releases"))
TOKEN = os.environ.get("RELOAD_TOKEN", "")
TRACE_EXPERIMENT = os.environ.get("TRACE_EXPERIMENT", "ch12-flower-support-serving")
CANARY = "주문번호 FL-000123 장미 꽃다발이 아직 안 왔어요. 내일 행사라 오늘 꼭 확인해 주세요."

client = mlflow.MlflowClient()
current: dict = {}


def chat(model: str, messages: list[dict], logprobs: bool = False) -> dict:
    r = httpx.post(f"{VLLM}/v1/chat/completions", timeout=60, json={
        "model": model, "messages": messages, "temperature": 0, "max_tokens": 256, "logprobs": logprobs,
        "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    return r.json()


def loaded_models() -> set[str]:
    return {m["id"] for m in httpx.get(f"{VLLM}/v1/models", timeout=10).json()["data"]}


def prepare(version: str) -> str:
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
    tuned = chat(lora_name, messages)["choices"][0]["message"]["content"]
    base = chat(BASE_MODEL, messages)["choices"][0]["message"]["content"]
    if tuned == base:
        raise HTTPException(500, f"{lora_name}의 출력이 베이스 모델과 같습니다. 어댑터가 적용되지 않았습니다.")
    if not check_output(CANARY, json.loads(tuned)).get("schema_valid"):
        raise HTTPException(500, f"{lora_name}의 출력이 형식에 맞지 않습니다: {tuned[:200]}")
    return {"model_version": model_version, "prompt_version": prompt_version, "lora_name": lora_name,
            "template": template}


def resolve() -> tuple[str, str]:
    mv = client.get_model_version_by_alias(MODEL, "champion")
    # 별칭으로 부른 프롬프트는 기본 60초 동안 캐시된다 — 방금 옮긴 별칭을 읽으려면 캐시를 끈다 (11-5)
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT}@production", cache_ttl_seconds=0)
    paired = mv.tags.get("prompt_uri", "").rsplit("/", 1)[-1]          # 게이트에서 함께 평가된 프롬프트 (09-5)
    if paired != str(prompt.version):
        raise HTTPException(409, f"champion v{mv.version}은 프롬프트 v{paired}와 평가되었는데 production은 v{prompt.version}입니다")
    return mv.version, str(prompt.version)


async def wait_for_vllm(timeout_s: int = 900):
    """vLLM이 준비될 때까지 기다린다. compose의 depends_on(service_healthy)은 docker compose up 때만 지켜지고,
    서버 재부팅 뒤 Docker가 컨테이너를 다시 띄울 때는 무시된다 — 기다리지 않으면 재시작을 반복한다 (12장 실측: 15번)."""
    async with httpx.AsyncClient() as http:
        for _ in range(timeout_s // 5):
            try:
                if (await http.get(f"{VLLM}/health", timeout=5)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
    raise RuntimeError(f"{VLLM}이 {timeout_s}초 안에 준비되지 않았습니다")


@asynccontextmanager
async def lifespan(_):
    await wait_for_vllm()
    mlflow.set_experiment(TRACE_EXPERIMENT)
    if os.environ.get("TRACING", "on") == "off":    # 기록 비용을 재려고 끌 수 있게 (11-2)
        mlflow.tracing.disable()
    current.update(activate(*resolve()))
    yield


app = FastAPI(title="flower-support gateway", lifespan=lifespan)


class Inquiry(BaseModel):
    inquiry: str
    channel: str = "web"          # 기능·경로별 사용량 집계에 쓴다 (11-1)


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": current.get("model_version"),
            "prompt_version": current.get("prompt_version")}


def record_metrics(snapshot: dict, channel: str, usage: dict, signals: dict, confidence: float | None,
                   result: dict | None, seconds: float):
    """요청 하나를 Prometheus 지표에 더한다 (12-1). 신호는 '문제가 있다' 쪽으로 센다."""
    v = (snapshot["model_version"], snapshot["prompt_version"])
    REQUESTS.labels(*v, channel, "ok" if result is not None else "invalid_json").inc()
    LATENCY.labels(*v).observe(seconds)
    TOKENS.labels(*v, "input").inc(usage["prompt_tokens"])
    TOKENS.labels(*v, "output").inc(usage["completion_tokens"])
    for name, bad in [("schema_invalid", not signals.get("schema_valid", False)),
                      ("order_id_ungrounded", not signals.get("order_id_grounded", True)),
                      ("product_not_in_inquiry", not signals.get("product_in_inquiry", True)),
                      ("product_not_in_catalog", not signals.get("product_in_catalog", True)),
                      ("pii_masked", bool(signals.get("pii_masked"))),
                      ("injection_suspected", signals.get("injection_suspected", False))]:
        if bad:
            SIGNALS.labels(*v, name).inc()
    if confidence is not None:
        CONFIDENCE.labels(*v).observe(confidence)


@app.middleware("http")
async def track_in_flight(request: Request, call_next):
    """/extract 요청을 받은 순간부터 답할 때까지 센다. 미들웨어는 이벤트 루프에서 돌므로 스레드 대기까지 잡힌다 (12-2)."""
    if request.url.path != "/extract":
        return await call_next(request)
    IN_FLIGHT.inc()
    start = time.perf_counter()
    try:
        return await call_next(request)
    finally:
        IN_FLIGHT.dec()
        TOTAL_LATENCY.observe(time.perf_counter() - start)


# 지표 경로는 async로 둔다: 동기 함수면 /extract와 같은 작업 스레드를 기다려야 해서,
# 요청이 몰릴 때 Prometheus의 읽기가 시간 초과로 실패하고 '서버 중단'으로 보인다 (12-2 실측)
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/vllm-metrics")
async def vllm_metrics():
    """vLLM은 127.0.0.1에만 열려 있으므로(10-2) 지표만 읽어 내보낸다. 어댑터를 올리는 API는 그대로 닫혀 있다."""
    async with httpx.AsyncClient() as http:
        r = await http.get(f"{VLLM}/metrics", timeout=10)
    return Response(r.content, media_type=r.headers.get("content-type", CONTENT_TYPE_LATEST))


@app.post("/extract")
def extract(req: Inquiry):
    snapshot = dict(current)
    start = time.perf_counter()
    inquiry, pii = mask_pii(req.inquiry)                   # 모델과 기록에는 가린 글만 간다
    with mlflow.start_span(name="extract", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"inquiry": inquiry})
        mlflow.update_current_trace(tags={"model_version": snapshot["model_version"],
                                          "prompt_version": snapshot["prompt_version"], "channel": req.channel})
        with mlflow.start_span(name="vllm", span_type=SpanType.CHAT_MODEL) as span:
            span.set_attributes({"model": snapshot["lora_name"], "prompt": f"{PROMPT}/{snapshot['prompt_version']}"})
            messages = snapshot["template"].format(inquiry=inquiry)
            try:
                resp = chat(snapshot["lora_name"], messages, logprobs=True)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:
                    raise
                prepare(snapshot["model_version"])            # vLLM 재시작으로 어댑터가 사라졌으면 다시 올린다 (10-5)
                resp = chat(snapshot["lora_name"], messages, logprobs=True)
            usage = resp["usage"]
            span.set_attribute("mlflow.chat.tokenUsage", {"input_tokens": usage["prompt_tokens"],
                                                          "output_tokens": usage["completion_tokens"],
                                                          "total_tokens": usage["total_tokens"]})
            text = resp["choices"][0]["message"]["content"]
            span.set_outputs({"content": text})
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = None
        checks = check_output(inquiry, result)
        confidence = category_confidence((resp["choices"][0].get("logprobs") or {}).get("content") or [])
        signals = {**checks, "pii_masked": pii, "injection_suspected": injection_suspected(inquiry),
                   "category_confidence": round(confidence, 4) if confidence is not None else None}
        root.set_attributes(signals)
        root.set_outputs({"result": result})
        request_id = root.trace_id
    record_metrics(snapshot, req.channel, usage, signals, confidence, result, time.perf_counter() - start)
    if result is None:
        raise HTTPException(502, f"모델 출력이 JSON이 아닙니다 (request_id {request_id})")
    return {"result": result, "model_version": snapshot["model_version"], "prompt_version": snapshot["prompt_version"],
            "request_id": request_id, "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


class Feedback(BaseModel):
    request_id: str
    correct: bool                          # 상담원이 본 결과가 맞았는가
    corrected: dict | None = None          # 틀렸다면 바로잡은 값 (category 등)
    comment: str = ""


@app.post("/feedback")
def feedback(fb: Feedback):
    """상담원의 판정을 그 요청의 트레이스에 남긴다. 바로잡은 값은 '기대 출력'으로 남겨 평가 세트로 되돌린다 (11-3, 11-5)."""
    mlflow.flush_trace_async_logging()   # 트레이스는 비동기로 올라간다 — 응답 직후의 피드백이 트레이스보다 먼저 도착할 수 있다
    source = AssessmentSource(source_type="HUMAN", source_id="상담원")
    mlflow.log_feedback(trace_id=fb.request_id, name="user_correct", value=fb.correct, rationale=fb.comment,
                        source=source)
    if fb.corrected:
        mlflow.log_expectation(trace_id=fb.request_id, name="expected_result", value=fb.corrected, source=source)
    return {"ok": True}


@app.post("/reload")
def reload(x_reload_token: str = Header(default="")):
    if not TOKEN or x_reload_token != TOKEN:
        raise HTTPException(403, "reload 토큰이 맞지 않습니다")
    before = {k: current.get(k) for k in ("model_version", "prompt_version")}
    start = time.perf_counter()
    new = activate(*resolve())
    current.update(new)
    return {"before": before, "after": {k: new[k] for k in ("model_version", "prompt_version")},
            "seconds": round(time.perf_counter() - start, 2)}
