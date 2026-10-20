"""합성 차량 번호판 데이터 (14-2). 실제 번호판 사진 대신, 글꼴로 그린 번호판을 무작위 배경에 붙인다.

    uv run python -m src.synth --font /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc --out data/scenes

- 장면(scene): 320×240 배경에 번호판 하나. 검출 모델의 입력과 정답 상자(bbox)
- 문자열: 승용차 번호 형식 "123가 4567"(3자리) 또는 "12가 3456"(2자리)
실제 서비스의 사진과는 거리가 멀다. 이 책의 목적은 번호판 인식 모델이 아니라 '모델 두 개로 된 서비스'의 운영이다.
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HANGUL = list("가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호")   # 승용차·대여 사업용
SCENE = (320, 240)


def plate_text(rng: random.Random) -> str:
    front = rng.randint(100, 999) if rng.random() < 0.5 else rng.randint(10, 99)
    return f"{front}{rng.choice(HANGUL)} {rng.randint(1000, 9999)}"


def render_plate(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    img = Image.new("RGB", (520, 110), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, 516, 106], outline="black", width=5)
    w = d.textlength(text, font=font)
    d.text(((520 - w) / 2, 4), text, fill="black", font=font)
    return img


def background(rng: random.Random) -> Image.Image:
    """도로·차체 대신 쓰는 무작위 배경: 그라데이션 + 사각형 몇 개 + 잡음."""
    c1, c2 = [tuple(rng.randint(30, 220) for _ in range(3)) for _ in range(2)]
    img = Image.new("RGB", SCENE)
    d = ImageDraw.Draw(img)
    for y in range(SCENE[1]):
        t = y / SCENE[1]
        d.line([(0, y), (SCENE[0], y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(c1, c2)))
    for _ in range(rng.randint(2, 6)):
        x, y = rng.randint(0, 300), rng.randint(0, 220)
        d.rectangle([x, y, x + rng.randint(20, 120), y + rng.randint(10, 80)],
                    fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return img


def scene(rng: random.Random, font) -> tuple[Image.Image, list[float], str]:
    text = plate_text(rng)
    plate = render_plate(text, font)
    width = rng.randint(90, 200)                               # 번호판 크기: 멀리·가까이
    plate = plate.resize((width, int(width * 110 / 520)))
    angle = rng.uniform(-8, 8)                                 # 기울기
    plate = plate.convert("RGBA").rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    img = background(rng)
    x = rng.randint(0, SCENE[0] - plate.width)
    y = rng.randint(0, SCENE[1] - plate.height)
    img.paste(plate, (x, y), plate)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    bbox = [x / SCENE[0], y / SCENE[1], (x + plate.width) / SCENE[0], (y + plate.height) / SCENE[1]]   # 0~1
    return img, [round(v, 4) for v in bbox], text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--font", required=True)
    p.add_argument("--out", default="data/scenes")
    p.add_argument("--n", type=int, nargs=3, default=[3000, 500, 500], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=14)
    args = p.parse_args()
    font = ImageFont.truetype(args.font, 84, index=1)          # NotoSansCJK .ttc의 1번 = 한국어(KR)
    for split, n, offset in zip(("train", "val", "test"), args.n, (0, 1, 2)):
        rng = random.Random(args.seed * 10 + offset)          # 세트마다 다른 시드 — 같은 번호판이 겹치지 않게
        out = Path(args.out) / split
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "labels.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for i in range(n):
                img, bbox, text = scene(rng, font)
                name = f"{i:05d}.jpg"
                img.save(out / name, quality=92)
                f.write(json.dumps({"file": name, "bbox": bbox, "text": text}, ensure_ascii=False) + "\n")
        print(f"{split}: {n}장")


if __name__ == "__main__":
    main()
