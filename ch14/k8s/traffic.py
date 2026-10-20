"""flowers-api에 꽃 사진을 계속 보내며 오류·버전·지연을 잰다 (14-4). 롤링 업데이트·자동 확장 중에 돌려 둔다.

    uv run python traffic.py --images data/raw --seconds 120 --workers 2 --out logs/rollout.jsonl

응답마다 (시각, 상태 코드, 모델 버전, 지연)을 남기고, 끝나면 버전별로 요약한다. 상태 0은 연결 자체가 실패한 것.
"""

import argparse
import itertools
import json
import threading
import time
from collections import Counter
from pathlib import Path

import httpx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:30080")
    p.add_argument("--images", required=True)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="")
    args = p.parse_args()
    files = [f.read_bytes() for f in sorted(Path(args.images).rglob("*.jpg"))[:300]]
    cycle, lock, records = itertools.cycle(files), threading.Lock(), []
    end = time.time() + args.seconds

    def worker():
        with httpx.Client(timeout=30) as client:
            while time.time() < end:
                with lock:
                    data = next(cycle)
                t = time.perf_counter()
                try:
                    r = client.post(f"{args.url}/predict", files={"file": ("f.jpg", data, "image/jpeg")})
                    status, version = r.status_code, r.json().get("model_version", "-") if r.status_code == 200 else "-"
                except httpx.HTTPError:
                    status, version = 0, "-"
                with lock:
                    records.append({"ts": round(time.time(), 3), "status": status, "version": version,
                                    "ms": round((time.perf_counter() - t) * 1000, 1)})

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    ms = sorted(r["ms"] for r in records)
    print(f"요청 {len(records)}건, 초당 {len(records) / args.seconds:.1f}건, 상태 {dict(Counter(r['status'] for r in records))}, "
          f"버전 {dict(Counter(r['version'] for r in records))}, p50 {ms[len(ms) // 2]:.0f}ms p99 {ms[int(len(ms) * 0.99)]:.0f}ms")


if __name__ == "__main__":
    main()
