"""번호판 모델 학습 (14-2). 실행 하나에 모델 하나. MLflow 실험 ch14-plates, 등록 모델 plate-detector / plate-recognizer.

    uv run python -m src.train detector --epochs 6
    uv run python -m src.train recognizer --epochs 20 --crops gt                          # 정답 상자로 자른 조각
    uv run python -m src.train recognizer --epochs 20 --crops detector --detector-version 1   # 검출 모델이 자른 조각

인식 모델의 입력은 검출 모델의 출력으로 정해진다. 정답 상자로 자른 깨끗한 조각으로만 배우면, 운영에서 검출 모델이
조금 빗나가게 자른 조각을 처음 보게 된다(14-2의 핵심 실험).
"""

import argparse
import hashlib
import os
import random
import subprocess
import time

import mlflow
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.models import CRNN, crop, decode, det_input, detector, encode, iou, load_split, rec_input, to_tensor

NAMES = {"detector": "plate-detector", "recognizer": "plate-recognizer"}


def code_version() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "."], capture_output=True, text=True).stdout)
    return {"git_commit": commit, "git_dirty": str(dirty)}


def data_version(root: str) -> str:
    return hashlib.md5(b"".join(open(f"{root}/{s}/labels.jsonl", "rb").read() for s in ("train", "val", "test"))).hexdigest()[:12]


def predict_boxes(version: str, rows: list[dict]) -> list[list[float]]:
    model = mlflow.pyfunc.load_model(f"models:/{NAMES['detector']}/{version}")
    out = []
    for i in range(0, len(rows), 64):
        x = np.stack([det_input(Image.open(r["path"])) for r in rows[i:i + 64]])
        out += np.asarray(model.predict(x)).tolist()
    return out


def train_detector(rows, val, args):
    model = detector()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X = to_tensor([det_input(Image.open(r["path"])) for r in rows])
    Y = torch.tensor([r["bbox"] for r in rows], dtype=torch.float32)
    Xv = to_tensor([det_input(Image.open(r["path"])) for r in val])
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.smooth_l1_loss(model(X[idx]), Y[idx], beta=0.02)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xv[i:i + 128]) for i in range(0, len(Xv), 128)]).tolist()
        ious = [iou(p, r["bbox"]) for p, r in zip(pred, val)]
        metrics = {"val_mean_iou": float(np.mean(ious)), "val_hit_rate": float(np.mean([v >= 0.5 for v in ious])),
                   "train_loss": loss.item()}
        mlflow.log_metrics(metrics, step=epoch)
        print(f"epoch {epoch + 1}: {metrics}")
    return model, Xv[:4], metrics


def train_recognizer(rows, val, args):
    if args.crops == "detector":                      # 검출 모델이 실제로 자른 조각으로 배운다
        boxes = predict_boxes(args.detector_version, rows)
    else:
        boxes = [r["bbox"] for r in rows]
    images = [Image.open(r["path"]).convert("RGB") for r in rows]
    targets = [torch.tensor(encode(r["text"])) for r in rows]
    Xv = to_tensor([rec_input(crop(Image.open(r["path"]), r["bbox"])) for r in val])
    model = CRNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            x = to_tensor([rec_input(crop(images[j], boxes[j], pad=rng.uniform(0.0, 0.08))) for j in idx])
            logp = model(x).permute(1, 0, 2)                                  # CTC는 (T, B, C)
            y = [targets[j] for j in idx]
            loss = ctc(logp, torch.cat(y), torch.full((len(idx),), logp.shape[0], dtype=torch.long),
                       torch.tensor([len(t) for t in y]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            texts = [decode(lp) for lp in model(Xv).numpy()]
        exact = float(np.mean([t == r["text"] for t, r in zip(texts, val)]))
        mlflow.log_metrics({"val_exact_gt_crops": exact, "train_loss": loss.item()}, step=epoch)
        print(f"epoch {epoch + 1}: exact {exact:.4f} loss {loss.item():.4f}")
    return model, Xv[:4], {"val_exact_gt_crops": exact}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["detector", "recognizer"])
    p.add_argument("--data", default="data/scenes")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--crops", choices=["gt", "detector"], default="gt")
    p.add_argument("--detector-version", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rows, val = load_split(args.data, "train"), load_split(args.data, "val")
    mlflow.set_experiment("ch14-plates")
    with mlflow.start_run(run_name=args.kind if args.kind == "detector" else f"recognizer-{args.crops}") as run:
        mlflow.log_params(vars(args))
        tags = code_version() | {"data_version": data_version(args.data), "kind": args.kind}
        if args.kind == "recognizer":
            tags |= {"crops": args.crops, "crops_from_detector": args.detector_version or "-"}
        mlflow.set_tags(tags)
        start = time.time()
        model, example, metrics = (train_detector if args.kind == "detector" else train_recognizer)(rows, val, args)
        mlflow.log_metric("train_minutes", (time.time() - start) / 60)
        model.eval()
        os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
        info = mlflow.pytorch.log_model(model.cpu(), name="model", input_example=example.numpy())   # 예시 4장 — 배치 크기 고정 방지 (02-5)
    mv = mlflow.register_model(info.model_uri, NAMES[args.kind],
                               tags=tags | {k: f"{v:.4f}" for k, v in metrics.items()})
    print(f"{NAMES[args.kind]} v{mv.version} 등록 (run {run.info.run_id[:8]}) {metrics}")


if __name__ == "__main__":
    main()
