"""동시 요청을 보내 처리량과 지연시간 분포를 잰다 (06-5).

실행:
    uv run python src/loadtest.py http://localhost:8000 --concurrency 8 --requests 400
"""

import argparse
import asyncio
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("--concurrency", type=int, default=8)
parser.add_argument("--requests", type=int, default=400)
args = parser.parse_args()

images = [p.read_bytes() for p in sorted(Path("data/raw").glob("*/*.jpg"))[:200]]


async def worker(client, queue, latencies, errors):
    while True:
        i = await queue.get()
        if i is None:
            return
        start = time.perf_counter()
        try:
            r = await client.post("/predict", files={"file": ("a.jpg", images[i % len(images)], "image/jpeg")})
            r.raise_for_status()
            latencies.append((time.perf_counter() - start) * 1000)
        except httpx.HTTPError:
            errors.append(i)


async def main():
    queue, latencies, errors = asyncio.Queue(), [], []
    for i in range(args.requests):
        queue.put_nowait(i)
    for _ in range(args.concurrency):
        queue.put_nowait(None)
    async with httpx.AsyncClient(base_url=args.url, timeout=60) as client:
        await client.get("/health")
        start = time.perf_counter()
        await asyncio.gather(*(worker(client, queue, latencies, errors) for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - start

    latencies.sort()

    def pct(q):
        return latencies[min(len(latencies) - 1, int(len(latencies) * q))]

    print(f"concurrency={args.concurrency} requests={args.requests} errors={len(errors)}")
    print(f"throughput {len(latencies) / elapsed:.1f} req/s | "
          f"p50 {pct(0.50):.0f}ms  p95 {pct(0.95):.0f}ms  p99 {pct(0.99):.0f}ms")


asyncio.run(main())
