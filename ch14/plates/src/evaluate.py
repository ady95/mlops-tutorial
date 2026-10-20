"""검출 + 인식을 이어 붙인 끝단(end-to-end) 평가 (14-2).

    uv run python -m src.evaluate --detector champion --recognizer champion
    uv run python -m src.evaluate --detector 1 --recognizer 2 --split test

모델 하나씩 재면 둘 다 좋아 보여도, 이어 붙이면 앞 모델의 작은 오차가 뒤 모델에서 커진다.
그래서 세 가지를 같이 잰다: 검출만(IoU), 인식만(정답 상자 조각), 둘을 이은 끝단(검출 상자 조각).
"""

import argparse
import json
import time

import mlflow
import numpy as np
from PIL import Image

from src.models import crop, decode, det_input, iou, load_split, rec_input

NAMES = {"detector": "plate-detector", "recognizer": "plate-recognizer"}


def model_uri(kind: str, ref: str) -> str:
    """'3' → 버전 3, 'champion' → 별칭."""
    return f"models:/{NAMES[kind]}/{ref}" if ref.isdigit() else f"models:/{NAMES[kind]}@{ref}"


def batched(model, arrays, size=64) -> np.ndarray:
    return np.concatenate([np.asarray(model.predict(np.stack(arrays[i:i + size]))) for i in range(0, len(arrays), size)])


def end_to_end(det_ref: str, rec_ref: str, rows: list[dict]) -> dict:
    det = mlflow.pyfunc.load_model(model_uri("detector", det_ref))
    rec = mlflow.pyfunc.load_model(model_uri("recognizer", rec_ref))
    images = [Image.open(r["path"]).convert("RGB") for r in rows]

    t = time.perf_counter()
    boxes = batched(det, [det_input(img) for img in images])
    det_ms = (time.perf_counter() - t) * 1000 / len(rows)
    ious = np.array([iou(b, r["bbox"]) for b, r in zip(boxes, rows)])

    gt_texts = [decode(lp) for lp in batched(rec, [rec_input(crop(img, r["bbox"])) for img, r in zip(images, rows)])]
    t = time.perf_counter()
    e2e_texts = [decode(lp) for lp in batched(rec, [rec_input(crop(img, b)) for img, b in zip(images, boxes)])]
    rec_ms = (time.perf_counter() - t) * 1000 / len(rows)

    truth = [r["text"] for r in rows]
    return {
        "det_mean_iou": float(ious.mean()),
        "det_hit_rate": float((ious >= 0.5).mean()),                                    # IoU 0.5 이상
        "rec_exact_gt_crops": float(np.mean([p == y for p, y in zip(gt_texts, truth)])),  # 인식만
        "e2e_exact": float(np.mean([p == y for p, y in zip(e2e_texts, truth)])),          # 끝단
        "det_ms_per_image": det_ms,
        "rec_ms_per_image": rec_ms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="champion")
    p.add_argument("--recognizer", default="champion")
    p.add_argument("--data", default="data/scenes")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    m = end_to_end(args.detector, args.recognizer, load_split(args.data, args.split))
    print(json.dumps({"detector": args.detector, "recognizer": args.recognizer, "split": args.split}
                     | {k: round(v, 4) for k, v in m.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
