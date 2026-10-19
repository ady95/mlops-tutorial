"""vLLM 부하 테스트: 동시 요청 수에 따른 처리량, 첫 토큰 지연(TTFT), 전체 지연 (10-4).

6장의 loadtest.py와 같은 방식(정해진 수의 작업자가 요청을 연달아 보냄)에 스트리밍을 더해 첫 토큰 시각을 잰다.

실행:
    uv run python -m src.bench_llm --model flower-support-v1 --concurrency 1 4 16 64
"""

import argparse
import asyncio
import json
import time
import urllib.request

import mlflow
from openai import AsyncOpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8001/v1")
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", default="prompts:/flower-support-extract@production")
parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
parser.add_argument("--requests", type=int, default=240)
args = parser.parse_args()

prompt = mlflow.genai.load_prompt(args.prompt)
inquiries = [json.loads(line)["inquiry"] for line in open("eval/holdout.jsonl", encoding="utf-8")]
client = AsyncOpenAI(base_url=args.base_url, api_key="unused", max_retries=0, timeout=300)


async def request(i: int) -> dict:
    start = time.perf_counter()
    first, tokens = None, 0
    stream = await client.chat.completions.create(
        model=args.model, messages=prompt.format(inquiry=inquiries[i % len(inquiries)]), temperature=0,
        max_tokens=256, stream=True, stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    async for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter()
        if chunk.usage:
            tokens = chunk.usage.completion_tokens
    end = time.perf_counter()
    return {"ttft": (first - start) * 1000, "latency": (end - start) * 1000, "tokens": tokens}


async def run(concurrency: int) -> dict:
    queue = asyncio.Queue()
    for i in range(args.requests):
        queue.put_nowait(i)
    results = []

    async def worker():
        while not queue.empty():
            results.append(await request(queue.get_nowait()))

    stop, peaks = asyncio.Event(), {}
    sampler = asyncio.create_task(sample_metrics(stop, peaks))
    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - start
    stop.set()
    await sampler

    def pct(key, q):
        values = sorted(r[key] for r in results)
        return values[min(len(values) - 1, int(len(values) * q))]

    return {"concurrency": concurrency, "req_per_s": len(results) / elapsed,
            "tok_per_s": sum(r["tokens"] for r in results) / elapsed,
            "ttft_p50": pct("ttft", 0.5), "ttft_p95": pct("ttft", 0.95),
            "lat_p50": pct("latency", 0.5), "lat_p95": pct("latency", 0.95), **peaks}


def vllm_metrics() -> dict:
    """vLLM이 내보내는 Prometheus 지표(/metrics)에서 실행 중·대기 요청 수와 KV 캐시 사용률을 읽는다."""
    text = urllib.request.urlopen(args.base_url.removesuffix("/v1") + "/metrics").read().decode()
    found = {}
    for line in text.splitlines():
        for key in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"):
            if line.startswith(key + "{"):
                found[key.split(":")[1]] = float(line.rsplit(" ", 1)[1])
    return found


async def sample_metrics(stop: asyncio.Event, peaks: dict):
    """부하가 걸린 동안 0.5초마다 지표를 읽어 최댓값을 남긴다."""
    while not stop.is_set():
        for k, v in (await asyncio.to_thread(vllm_metrics)).items():
            peaks[k] = max(peaks.get(k, 0), v)
        await asyncio.sleep(0.5)


async def main():
    await request(0)                       # 첫 요청의 준비 시간을 빼기 위한 예열
    print(f"{args.model}: 요청 {args.requests}건씩")
    print(f"{'동시':>4} {'요청/초':>8} {'토큰/초':>8} {'TTFT p50':>9} {'p95':>7} {'지연 p50':>9} {'p95':>7}"
          f" {'최대 실행':>8} {'최대 대기':>8} {'KV 최대':>7}")
    for c in args.concurrency:
        r = await run(c)
        print(f"{c:>4} {r['req_per_s']:>8.2f} {r['tok_per_s']:>8.0f} {r['ttft_p50']:>7.0f}ms {r['ttft_p95']:>5.0f}ms "
              f"{r['lat_p50']:>7.0f}ms {r['lat_p95']:>5.0f}ms {r.get('num_requests_running', 0):>8.0f}"
              f" {r.get('num_requests_waiting', 0):>8.0f} {r.get('kv_cache_usage_perc', 0):>6.1%}")


asyncio.run(main())
