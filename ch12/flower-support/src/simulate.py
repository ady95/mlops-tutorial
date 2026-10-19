"""운영 트래픽 흉내: 평소 구간과, 새 상품 문의가 섞여 들어오는 구간을 차례로 보낸다 (11-1, 11-3, 11-4).

문의 목록은 simulate_data.py (12장 load.py와 함께 쓴다).
일부 문의에는 고객이 남긴 연락처를 덧붙여 민감정보 가리기가 작동하는지 본다.

실행:
    uv run python -m src.simulate --normal 300 --drift 160
"""

import argparse
import asyncio
import json
import random
import time

import httpx

from src.simulate_data import CHANNELS, drift, normal, with_contact

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8002")
parser.add_argument("--normal", type=int, default=300, help="평소 구간 요청 수")
parser.add_argument("--drift", type=int, default=160, help="새 상품 문의가 절반 섞인 구간 요청 수")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

rng = random.Random(7)


async def send(client, queue, log):
    while not queue.empty():
        phase, text = queue.get_nowait()
        r = await client.post(f"{args.url}/extract", timeout=60,
                              json={"inquiry": with_contact(text, rng), "channel": rng.choice(CHANNELS)})
        log.append({"phase": phase, "ok": r.status_code == 200,
                    "request_id": r.json().get("request_id") if r.status_code == 200 else None})


async def run_phase(name: str, items: list[str]) -> dict:
    queue = asyncio.Queue()
    for text in items:
        queue.put_nowait((name, text))
    log, start = [], time.time()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(send(client, queue, log) for _ in range(args.workers)))
    end = time.time()
    print(f"[{name}] {len(log)}건, 실패 {sum(not x['ok'] for x in log)}건, {end - start:.0f}초")
    return {"phase": name, "start_ms": int(start * 1000), "end_ms": int(end * 1000), "n": len(log)}


async def main():
    phases = [await run_phase("normal", rng.sample(normal, args.normal))]
    mixed = rng.sample(drift, min(len(drift), args.drift // 2)) + rng.sample(normal, args.drift // 2)
    rng.shuffle(mixed)
    phases.append(await run_phase("drift", mixed))
    json.dump(phases, open("eval/outputs/simulate_phases.json", "w", encoding="utf-8"), indent=2)
    print("구간 기록 → eval/outputs/simulate_phases.json")


asyncio.run(main())
