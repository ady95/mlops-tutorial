"""운영 보조 에이전트의 뼈대 (13장).

에이전트 = "모델이 도구를 골라 부르고, 결과를 보고 다시 판단하는 반복"이다. 이 파일이 그 반복 전부다.
- 도구는 함수 + JSON 스키마. 모델은 스키마만 보고 무엇을 부를지 정한다.
- 도구 결과는 모델에 보내기 전에 지운다(sanitize): 내부 주소·계정·키가 외부 API로 나가지 않게.
- 한 번의 실행 전체를 MLflow 트레이스로 남긴다 (11장). 에이전트도 관측 대상이다.

환경변수: OPENAI_API_KEY, OPENAI_MODEL (OpenAI 호환 서버면 OPENAI_BASE_URL), MLFLOW_TRACKING_URI
"""

import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

from agent.sanitize import sanitize

MAX_RESULT_CHARS = 6000          # 도구 결과가 길면 자른다 — 모델 입력 비용과 혼란을 줄인다


@dataclass
class Tool:
    fn: Callable
    description: str
    params: dict                 # JSON 스키마의 properties
    required: list = field(default_factory=list)
    # 무언가를 바꾸는 도구인가 (13-5: 바꾸는 도구는 '제안'만 한다)
    writes: bool = False

    @property
    def name(self) -> str:
        return self.fn.__name__

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": self.required}}}


def tool(description: str, writes: bool = False, **params):
    """함수를 도구로 만든다. params: 인자 이름 → JSON 스키마. 기본값이 없는 인자는 필수."""
    def wrap(fn):
        sig = inspect.signature(fn)
        required = [n for n, p in sig.parameters.items() if p.default is inspect.Parameter.empty]
        return Tool(fn, description, params, required, writes)
    return wrap


def call_tool(t: Tool, arguments: str) -> str:
    try:
        result = t.fn(**json.loads(arguments or "{}"))
    except Exception as e:          # 도구 오류도 모델에게 그대로 알려 다음 행동을 고르게 한다
        result = f"도구 오류: {type(e).__name__}: {e}"
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    text = sanitize(text)
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + f"\n...(잘림, 전체 {len(text)}자)"


def run(task: str, tools: list[Tool], system: str, max_steps: int = 15, name: str = "agent") -> dict:
    """task를 끝낼 때까지(모델이 도구 없이 답할 때까지) 반복한다. max_steps를 넘으면 멈춘다."""
    client = OpenAI()
    model = os.environ["OPENAI_MODEL"]
    by_name = {t.name: t for t in tools}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    calls, usage, start = [], {"input": 0, "output": 0}, time.time()
    with mlflow.start_span(name=name, span_type=SpanType.AGENT) as root:
        root.set_inputs({"task": task})
        for step in range(1, max_steps + 1):
            resp = client.chat.completions.create(model=model, messages=messages, tools=[t.spec() for t in tools])
            usage["input"] += resp.usage.prompt_tokens
            usage["output"] += resp.usage.completion_tokens
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                answer = msg.content or ""
                break
            for c in msg.tool_calls:
                t = by_name.get(c.function.name)
                with mlflow.start_span(name=c.function.name, span_type=SpanType.TOOL) as span:
                    span.set_inputs(c.function.arguments)
                    out = call_tool(t, c.function.arguments) if t else f"없는 도구: {c.function.name}"
                    span.set_outputs(out)
                calls.append({"step": step, "tool": c.function.name, "args": c.function.arguments, "chars": len(out)})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        else:
            answer = None                      # 끝내지 못했다 — 사람에게 넘긴다
        result = {"answer": answer, "steps": step, "calls": calls, "tokens": usage,
                  "seconds": round(time.time() - start, 1), "trace_id": root.trace_id, "messages": messages}
        root.set_outputs({"answer": answer, "steps": step, "tool_calls": len(calls)})
    return result
