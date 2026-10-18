"""MLOps 따라하기 02-5: Fashion-MNIST 분류 모델을 학습하고 MLflow에 기록한다.

실행:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run train.py --epochs 3
"""

import argparse
import os
import platform
import time

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_set = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=512, num_workers=2)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("ch02-first-run")
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args) | {"device": device, "model": "SmallCNN"})
        mlflow.set_tags({
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "none",
        })

        start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            test_loss, test_acc = evaluate(model, test_loader, device)
            mlflow.log_metrics({"train_loss": loss.item(), "test_loss": test_loss, "test_accuracy": test_acc}, step=epoch)
            print(f"epoch {epoch}: test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")

        mlflow.log_metric("train_seconds", time.time() - start)

        # MLflow 3는 PyTorch 모델을 torch.export(pt2) 형식으로 저장하며, 저장한 장치가 모델에 고정된다.
        # CPU로 옮겨 저장하고, 저장 직후 검증도 CPU에서 하도록 지정한다 (GPU 서버에서 실행할 때 필요).
        os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
        # 입력 예시의 배치 크기가 1이면 모델이 배치 1 전용으로 고정되므로 여러 장을 넘긴다.
        example = next(iter(test_loader))[0][:4].numpy()
        mlflow.pytorch.log_model(model.cpu(), name="model", input_example=example)
        print(f"run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
