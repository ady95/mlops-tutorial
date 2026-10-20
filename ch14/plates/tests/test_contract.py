"""두 모델 사이의 계약(자르기·부호화)을 지키는지 — 모델 없이 도는 단위 테스트 (14-2)."""

import random

import numpy as np
from PIL import Image

from app.main import PLATE
from src.models import CHARSET, REC_SIZE, crop, decode, encode, iou, rec_input
from src.synth import plate_text


def one_hot_path(text: str, steps: int = 32) -> np.ndarray:
    """문자열을 CTC 출력처럼 만든다: 문자마다 2칸 + 빈칸 1칸."""
    logp = np.full((steps, len(CHARSET) + 1), -10.0, dtype=np.float32)
    t = 0
    for k in encode(text):
        logp[t:t + 2, k] = 0
        logp[t + 2, 0] = 0
        t += 3
    logp[t:, 0] = 0
    return logp


def test_encode_decode_roundtrip():
    rng = random.Random(0)
    for _ in range(50):
        text = plate_text(rng)
        assert decode(one_hot_path(text)) == text
        assert PLATE.match(text)


def test_crop_shape_and_range():
    img = Image.new("RGB", (320, 240), "white")
    piece = crop(img, [0.9, 0.9, 1.0, 1.0])             # 가장자리 상자도 그림 밖으로 나가지 않는다
    assert piece.size == REC_SIZE
    x = rec_input(piece)
    assert x.shape == (1, 32, 128) and x.min() >= -1 and x.max() <= 1


def test_iou():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    assert iou([0, 0, 0.5, 1], [0.5, 0, 1, 1]) == 0
    assert abs(iou([0, 0, 1, 1], [0, 0, 0.5, 1]) - 0.5) < 1e-9
