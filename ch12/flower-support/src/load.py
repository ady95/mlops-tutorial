"""운영 트래픽 흉내 — 일정한 속도로 계속 보낸다 (12-1, 12-2, 12-3).

11장 simulate.py는 정해진 건수를 최대한 빨리 보냈다. 지표는 시간에 따른 변화를 보므로, 여기서는
초당 요청 수(--rps)를 정해 두고 도착 간격을 무작위로 하여 실제 서비스처럼 보낸다.
응답을 기다리지 않고 다음 요청을 보내므로, 처리 능력보다 빨리 보내면 vLLM 대기열이 쌓인다.

    uv run python -m src.load --minutes 30 --rps 1                       # 평소
    uv run python -m src.load --minutes 30 --rps 1 --drift 0.5           # 새 상품 문의가 절반
    uv run python -m src.load --minutes 5 --rps 40 --name spike          # 몰림
구간 기록은 eval/outputs/load_phases.json에 더해진다 (drift_llm.py가 읽는다).
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from src.simulate_data import CHANNELS, drift, normal, with_contact

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8002")
parser.add_argument("--minutes", type=float, default=10)
parser.add_argument("--rps", type=float, default=1)
parser.add_argument("--drift", type=float, default=0.0, help="새 상품 문의의 비율")
parser.add_argument("--name", default=None, help="구간 이름 (기본: normal 또는 drift)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
rng = random.Random(args.seed)
stats = {"ok": 0, "fail": 0}


async def one(client: httpx.AsyncClient, text: str):
    try:
        r = await client.post(f"{args.url}/extract", timeout=120,
                              json={"inquiry": with_contact(text, rng), "channel": rng.choice(CHANNELS)})
        stats["ok" if r.status_code == 200 else "fail"] += 1
    except httpx.HTTPError:
        stats["fail"] += 1


async def main():
    name = args.name or ("drift" if args.drift else "normal")
    start = time.time()
    tasks = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as client:
        while time.time() - start < args.minutes * 60:
            text = rng.choice(drift) if rng.random() < args.drift else rng.choice(normal)
            tasks.append(asyncio.create_task(one(client, text)))
            await asyncio.sleep(rng.expovariate(args.rps))
        await asyncio.gather(*tasks)
    end = time.time()
    print(f"[{name}] {len(tasks)}건 (성공 {stats['ok']}, 실패 {stats['fail']}), {end - start:.0f}초")
    path = Path("eval/outputs/load_phases.json")
    phases = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    phases.append({"phase": name, "start_ms": int(start * 1000), "end_ms": int(end * 1000), "n": len(tasks),
                   "rps": args.rps, "drift": args.drift})
    path.write_text(json.dumps(phases, ensure_ascii=False, indent=2), encoding="utf-8")


asyncio.run(main())
