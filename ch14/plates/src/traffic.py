"""plate API에 테스트 세트 사진을 계속 보내며 정확도·오류·버전을 잰다 (14-2). 모델 교체 중에도 돌려 둔다.

    uv run python -m src.traffic --seconds 60 --workers 2 --out logs/traffic.jsonl

응답마다 (시각, 상태 코드, 정답 여부, 검출·인식 버전, 지연)을 한 줄씩 남기고, 끝나면 버전 조합별로 요약한다.
"""

import argparse
import itertools
import json
import threading
import time
from collections import defaultdict

import httpx
import numpy as np

from src.models import load_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8010")
    p.add_argument("--data", default="data/scenes")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="")
    args = p.parse_args()
    rows = load_split(args.data, "test")
    files = [(r, open(r["path"], "rb").read()) for r in rows]
    cycle, lock, records = itertools.cycle(files), threading.Lock(), []
    end = time.time() + args.seconds

    def worker():
        with httpx.Client(timeout=30) as client:
            while time.time() < end:
                with lock:
                    row, data = next(cycle)
                t = time.perf_counter()
                try:
                    r = client.post(f"{args.url}/predict", files={"file": ("p.jpg", data, "image/jpeg")})
                    body = r.json() if r.status_code == 200 else {}
                    status = r.status_code
                except httpx.HTTPError:
                    body, status = {}, 0                     # 연결 자체가 실패
                v = body.get("versions", {})
                rec = {"ts": round(time.time(), 3), "status": status, "ok": body.get("text") == row["text"],
                       "pair": f"d{v.get('detector', '-')}/r{v.get('recognizer', '-')}",
                       "ms": round((time.perf_counter() - t) * 1000, 1)}
                with lock:
                    records.append(rec)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    by_pair = defaultdict(list)
    for r in records:
        by_pair[r["pair"]].append(r)
    errors = sum(r["status"] != 200 for r in records)
    print(f"요청 {len(records)}건, 오류 {errors}건, 초당 {len(records) / args.seconds:.1f}건")
    for pair, rs in by_pair.items():
        ms = [r["ms"] for r in rs]
        print(f"  {pair}: {len(rs)}건, 정확도 {np.mean([r['ok'] for r in rs]):.4f}, "
              f"지연 p50 {np.percentile(ms, 50):.1f}ms p95 {np.percentile(ms, 95):.1f}ms")


if __name__ == "__main__":
    main()
