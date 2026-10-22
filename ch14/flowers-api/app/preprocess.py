"""서비스용 전처리 — 학습 때의 처리를 그대로 재현한다 (06-1).

학습 데이터는 3장 prepare.py가 짧은 변을 맞추고 가운데를 잘라 224×224로 만들었고,
4장 train.py가 ImageNet 값으로 정규화했다. 서비스도 정확히 같은 순서로 처리해야 한다.
"""

import io

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# 학습 때 ImageFolder의 순서(이름순)
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def to_tensor(image_bytes: bytes) -> np.ndarray:
    """JPEG/PNG 바이트 → 모델 입력 (1, 3, 224, 224) float32."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.fit(img.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0          # (224, 224, 3), 0~1
    x = (x - MEAN) / STD                                        # ImageNet 정규화
    return x.transpose(2, 0, 1)[None]                           # (1, 3, 224, 224)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
