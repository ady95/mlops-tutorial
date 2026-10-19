"""드리프트 검사용 입력 특성 (12-3).

사진 자체는 남기지 않고, 분포를 비교할 수 있는 숫자 몇 개만 남긴다.
학습 파이프라인의 기준 데이터(src/reference.py)와 추론 서버(app/features.py)가 같은 함수를 쓴다 — 두 곳의 계산이 다르면
드리프트가 아닌데 드리프트로 보인다. 고칠 때는 두 파일을 함께 고친다.
"""

import numpy as np


def image_features(rgb: np.ndarray) -> dict:
    """224×224×3, 0~1 범위의 RGB 배열 → 밝기·대비·색온도."""
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return {"brightness": round(float(rgb.mean()), 4),                       # 평균 밝기
            "contrast": round(float(luma.std()), 4),                         # 밝기의 표준편차
            "warmth": round(float(rgb[..., 0].mean() - rgb[..., 2].mean()), 4)}   # 빨강 - 파랑 (조명 색)
