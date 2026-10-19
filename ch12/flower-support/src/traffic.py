"""업데이트·롤백 중에도 요청을 계속 보내, 응답한 버전과 실패를 시간순으로 기록한다 (10-5).

실행:
    uv run python -m src.traffic --seconds 120 --workers 4
"""

import argparse
import asyncio
import json
import time
from collections import Counter

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8002")
parser.add_argument("--seconds", type=int, default=120)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

inquiries = [json.loads(line)["inquiry"] for line in open("eval/holdout.jsonl", encoding="utf-8")]
log: list[dict] = []


async def worker(client: httpx.AsyncClient, wid: int, start: float):
    i = wid
    while time.perf_counter() - start < args.seconds:
        t = time.perf_counter()
        try:
            r = await client.post(f"{args.url}/extract", json={"inquiry": inquiries[i % len(inquiries)]}, timeout=60)
            ok = r.status_code == 200
            version = r.json().get("model_version") if ok else None
            prompt = r.json().get("prompt_version") if ok else None
        except httpx.HTTPError:
            ok, version, prompt = False, None, None
        log.append({"t": round(t - start, 2), "ok": ok, "model": version, "prompt": prompt,
                    "ms": round((time.perf_counter() - t) * 1000)})
        i += args.workers


async def main():
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(worker(client, w, start) for w in range(args.workers)))
    log.sort(key=lambda x: x["t"])
    print(f"요청 {len(log)}건, 실패 {sum(not x['ok'] for x in log)}건")
    print("버전별 응답:", dict(Counter((x["model"], x["prompt"]) for x in log if x["ok"])))
    prev = None
    for x in log:   # 응답 버전이 바뀐 순간을 찍는다
        key = (x["model"], x["prompt"])
        if x["ok"] and key != prev:
            print(f"  {x['t']:7.2f}초부터 모델 v{key[0]} + 프롬프트 v{key[1]}")
            prev = key
    ms = sorted(x["ms"] for x in log if x["ok"])
    print(f"지연 p50 {ms[len(ms) // 2]}ms, p95 {ms[int(len(ms) * 0.95)]}ms, 최대 {ms[-1]}ms")
    json.dump(log, open("eval/outputs/traffic.json", "w", encoding="utf-8"), ensure_ascii=False)


asyncio.run(main())
