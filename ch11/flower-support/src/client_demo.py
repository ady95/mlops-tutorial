"""OpenAI SDK로 vLLM 부르기: 일반 호출, 스트리밍 (10-2).

기존 OpenAI 코드는 환경변수만 바꾸면 그대로 로컬 모델을 부른다:
    export OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=unused
    uv run python -m src.client_demo
"""

import time

import mlflow
from openai import OpenAI

# 주소와 키는 OPENAI_BASE_URL, OPENAI_API_KEY 환경변수에서 읽는다
client = OpenAI()
print("모델 목록:", [m.id for m in client.models.list()])

prompt = mlflow.genai.load_prompt("prompts:/flower-support-extract@production")
messages = prompt.format(inquiry="주문번호 FL-031244 해바라기 꽃다발 받는 날짜를 다음 주로 바꾸고 싶어요. 급하지 않아요.")
# Qwen3.5 생각 모드 끄기 (08-3)
extra = {"chat_template_kwargs": {"enable_thinking": False}}

# 1. 일반 호출: 응답 전체를 한 번에 받는다
start = time.perf_counter()
resp = client.chat.completions.create(model="flower-support-adapter-v1", messages=messages,
                                      temperature=0, max_tokens=256, extra_body=extra)
print(f"\n[일반] {time.perf_counter() - start:.2f}초, 입력 {resp.usage.prompt_tokens}토큰 → 출력 {resp.usage.completion_tokens}토큰")
print(resp.choices[0].message.content)

# 2. 스트리밍: 토큰이 만들어지는 대로 조각(chunk)으로 받는다
start, first, pieces = time.perf_counter(), None, 0
stream = client.chat.completions.create(model="flower-support-adapter-v1", messages=messages,
                                        temperature=0, max_tokens=256, stream=True, extra_body=extra)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        first = first or time.perf_counter()
        pieces += 1
print(f"\n[스트리밍] 첫 조각 {(first - start) * 1000:.0f}ms, 마지막 조각 {(time.perf_counter() - start) * 1000:.0f}ms, "
      f"조각 {pieces}개")
