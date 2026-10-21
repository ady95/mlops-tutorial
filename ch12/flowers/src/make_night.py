"""'야간·실내 사진'이라는 새 촬영 조건을 흉내 낸다 (12-5).

실제로는 새 조건에서 찍힌 사진을 모으고 사람이 라벨을 붙인다. 이 책에서는 원본 사진을 어둡게(노출 15~35%),
따뜻한 조명색으로, 센서 잡음을 더해 만든다. 원본의 라벨을 그대로 쓴다.

    uv run python -m src.make_night --fraction 0.25      # data/raw/<클래스>/night_<원본이름>.jpg 생성

주의: 어둡게 만든 사진은 원본과 '같은 사진'이다. 원본이 테스트 세트에 있는데 어두운 사본이 학습 세트에 들어가면
테스트 점수가 부풀려진다. prepare.py가 night_ 사본을 원본과 같은 세트로 보낸다 (03-4의 근접 중복 묶음과 같은 원리).
"""

import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

PREFIX = "night_"


def to_night(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    x = x * rng.uniform(0.15, 0.35)                       # 노출 부족
    x = x * np.array([1.0, 0.85, 0.65])                   # 백열등 같은 따뜻한 조명
    x = x + rng.normal(0, rng.uniform(0.01, 0.03), x.shape)   # 어두울수록 두드러지는 센서 잡음
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))


def seed_of(name: str) -> int:
    return zlib.crc32(name.encode())                      # 같은 사진은 언제 만들어도 같은 결과


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    validation = Path("data/interim/validation.json")
    if not validation.exists():
        raise SystemExit("먼저 dvc repro로 품질 검사를 돌리세요 — 검사에서 제외된 원본은 사본을 만들지 않습니다")
    excluded = set(json.load(open(validation))["excluded"])    # 라벨 충돌·중복·손상 — 사본으로 되살리지 않는다
    rng = random.Random(args.seed)
    total = skipped = 0
    for class_dir in sorted(p for p in Path("data/raw").iterdir() if p.is_dir()):
        originals = sorted(p for p in class_dir.glob("*.jpg") if not p.name.startswith(PREFIX))
        for path in rng.sample(originals, round(len(originals) * args.fraction)):   # 고르는 방식은 그대로
            if f"{class_dir.name}/{path.name}" in excluded:
                skipped += 1
                continue
            with Image.open(path) as img:
                to_night(img, seed_of(path.name)).save(class_dir / f"{PREFIX}{path.name}", quality=90)
            total += 1
        print(f"{class_dir.name}: {round(len(originals) * args.fraction)}장 골라 만듦")
    print(f"합계 {total}장 (검사에서 제외된 원본 {skipped}장은 건너뜀)")
