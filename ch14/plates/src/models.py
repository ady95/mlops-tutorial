"""번호판 검출·인식 모델과 데이터 (14-2). 두 모델은 서로 다른 MLflow 모델로 등록하고 따로 배포한다.

- 검출(detector): 장면 사진 → 번호판 상자(x1, y1, x2, y2, 0~1). MobileNetV3-Small 전이학습 + 좌표 회귀
- 인식(recognizer): 번호판 조각(32×128 흑백) → 문자열. 작은 CRNN + CTC
두 모델 사이의 약속(계약): 인식 모델은 '검출 상자로 잘라 32×128로 맞춘 흑백 조각'을 받는다.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from src.synth import HANGUL

CHARSET = list("0123456789") + HANGUL           # CTC 빈칸(blank)은 0번, 문자는 1번부터
DET_SIZE = (224, 224)
REC_SIZE = (128, 32)                              # (가로, 세로)
MEAN, STD = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)


def detector(pretrained: bool = True) -> nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 4)
    return nn.Sequential(m, nn.Sigmoid())


class CRNN(nn.Module):
    def __init__(self, n_classes: int = len(CHARSET) + 1):
        super().__init__()
        def block(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool)]
        self.cnn = nn.Sequential(*block(1, 32, 2), *block(32, 64, 2), *block(64, 128, (2, 1)), *block(128, 128, (2, 1)))
        self.rnn = nn.LSTM(128 * 2, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):                             # x: (B, 1, 32, 128) → (B, 32, n_classes) 로그 확률
        f = self.cnn(x)                               # (B, 128, 2, 32)
        b, c, h, w = f.shape
        out, _ = self.rnn(f.permute(0, 3, 1, 2).reshape(b, w, c * h))
        return self.fc(out).log_softmax(-1)


def det_input(img: Image.Image) -> np.ndarray:
    x = np.asarray(img.convert("RGB").resize(DET_SIZE), dtype=np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def crop(img: Image.Image, bbox, pad: float = 0.04) -> Image.Image:
    """상자(0~1)로 잘라 인식 모델 입력 크기로. 상자보다 조금 넓게 자른다."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    box = (max(0, (x1 - pad * bw) * w), max(0, (y1 - pad * bh) * h), min(w, (x2 + pad * bw) * w), min(h, (y2 + pad * bh) * h))
    return img.crop(box).convert("L").resize(REC_SIZE)


def rec_input(piece: Image.Image) -> np.ndarray:
    return (np.asarray(piece, dtype=np.float32)[None] / 255 - 0.5) / 0.5


def encode(text: str) -> list[int]:
    return [CHARSET.index(ch) + 1 for ch in text.replace(" ", "")]


def decode(logp: np.ndarray) -> str:
    """CTC 탐욕 해독: 연속된 같은 기호와 빈칸을 지운다. 표시 형식 '123가 4567'로 띄운다."""
    best, out, prev = logp.argmax(-1), [], 0
    for k in best:
        if k != prev and k != 0:
            out.append(CHARSET[k - 1])
        prev = k
    s = "".join(out)
    i = next((n for n, ch in enumerate(s) if ch in HANGUL), None)
    return f"{s[:i + 1]} {s[i + 1:]}" if i is not None else s


def load_split(root: str, split: str) -> list[dict]:
    d = Path(root) / split
    return [json.loads(line) | {"path": str(d / json.loads(line)["file"])}
            for line in open(d / "labels.jsonl", encoding="utf-8")]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_tensor(arrays) -> torch.Tensor:
    return torch.from_numpy(np.stack(arrays))
