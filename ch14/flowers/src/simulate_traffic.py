"""운영 트래픽 흉내 (12-3, 12-5). 테스트 세트 사진을 추론 서버에 계속 보낸다.

    uv run python -m src.simulate_traffic --minutes 60 --rps 2                 # 평소
    uv run python -m src.simulate_traffic --minutes 60 --rps 2 --night 0.5     # 저녁 주문이 늘어난 뒤

--night 비율만큼은 make_night.py와 같은 방식으로 어둡게 만들어 보낸다. 테스트 세트(data/prepared/test)의
원본 사진만 쓰므로, 학습에 쓰인 사진이 섞이지 않는다.
"""

import argparse
import io
import random
import time
from pathlib import Path

import httpx
from PIL import Image

from src.make_night import PREFIX, seed_of, to_night

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
parser.add_argument("--minutes", type=float, default=10)
parser.add_argument("--rps", type=float, default=2)
parser.add_argument("--night", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

rng = random.Random(args.seed)
photos = sorted(p for p in Path("data/prepared/test").glob("*/*.jpg") if not p.name.startswith(PREFIX))
end = time.time() + args.minutes * 60
sent = night = errors = 0
with httpx.Client(base_url=args.url, timeout=30) as client:
    while time.time() < end:
        path = rng.choice(photos)
        img = Image.open(path).convert("RGB")
        if rng.random() < args.night:
            img = to_night(img, seed_of(path.name))
            night += 1
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        try:
            client.post("/predict", files={"file": (path.name, buf.getvalue(), "image/jpeg")}).raise_for_status()
        except httpx.HTTPError:
            errors += 1
        sent += 1
        time.sleep(max(0.0, rng.expovariate(args.rps)))   # 요청 간격을 무작위로 (평균 1/rps초)
print(f"보냄 {sent}건 (어두운 사진 {night}건), 오류 {errors}건")
