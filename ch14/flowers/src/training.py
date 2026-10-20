"""꽃 분류 모델 학습 — 4장 train.py를 파이프라인에서 부를 수 있는 함수로 정리했다 (07-2)."""

import os
import platform
import random
import subprocess
import time

import matplotlib
import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

matplotlib.use("Agg")

MODELS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(name, num_classes, freeze_backbone, pretrained=True):
    fn, weights = MODELS[name]
    model = fn(weights=weights if pretrained else None)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def data_version(lock_path="dvc.lock"):
    """DVC가 기록한 원본·학습 데이터의 해시."""
    lock = yaml.safe_load(open(lock_path))
    outs = {o["path"]: o["md5"] for stage in lock["stages"].values() for o in stage["outs"]}
    raw = next(d["md5"] for d in lock["stages"]["validate"]["deps"] if d["path"] == "data/raw")
    return {"data_raw_md5": raw, "data_prepared_md5": outs["data/prepared"]}


def code_version():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "."], capture_output=True, text=True).stdout)
    return {"git_commit": commit, "git_dirty": str(dirty)}


def run_epoch(model, loader, device, criterion, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, correct, total, preds, labels = 0.0, 0, 0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())
    return loss_sum / total, correct / total, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def cpu_latency_ms(model, size, runs=50):
    model = model.cpu().eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def train(config: dict, experiment: str, extra_tags: dict | None = None) -> dict:
    """config: model, epochs, batch_size, lr, freeze_backbone, seed, deterministic, num_workers."""
    set_seed(config["seed"], config["deterministic"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = yaml.safe_load(open("params.yaml"))["prepare"]["image_size"]

    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                   transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_set = datasets.ImageFolder("data/prepared/train", transform=train_tf)
    val_set = datasets.ImageFolder("data/prepared/val", transform=eval_tf)
    g = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, generator=g,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=64, num_workers=config["num_workers"])
    classes = train_set.classes

    model = build_model(config["model"], len(classes), config["freeze_backbone"]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(config | {"image_size": size, "device": device,
                                    "trainable_params": sum(p.numel() for p in trainable)})
        mlflow.set_tags(code_version() | data_version() | (extra_tags or {}) | {
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "none"})

        start = time.time()
        for epoch in range(1, config["epochs"] + 1):
            t = time.time()
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc, preds, labels = run_epoch(model, val_loader, device, criterion)
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss,
                                "val_accuracy": val_acc, "epoch_seconds": time.time() - t}, step=epoch)
            print(f"epoch {epoch}: train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        f1 = f1_score(labels, preds, average=None)
        metrics = {"train_seconds": time.time() - start, "val_accuracy": val_acc,
                   "val_f1_macro": float(f1.mean())}
        mlflow.log_metrics(metrics | {f"val_f1_{c}": float(v) for c, v in zip(classes, f1)})
        disp = ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=classes)
        disp.plot(cmap="Blues", colorbar=False)
        mlflow.log_figure(disp.figure_, "confusion_matrix_val.png")
        metrics["cpu_latency_ms"] = cpu_latency_ms(model, size)
        mlflow.log_metric("cpu_latency_ms", metrics["cpu_latency_ms"])

        os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
        example = next(iter(val_loader))[0][:4].numpy()
        mlflow.pytorch.log_model(model, name="model", input_example=example)
    return {"run_id": run.info.run_id} | metrics
