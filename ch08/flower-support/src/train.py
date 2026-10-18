"""LoRA·QLoRA 지도 미세조정 (08-3) + 체크포인트 (08-4) + MLflow 기록 (08-5).

실행:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run python -m src.train                        # params.yaml의 train 설정
    uv run python -m src.train --set lora_r=64        # 설정 하나만 바꿔 실험
    uv run python -m src.train --resume               # 마지막 체크포인트에서 이어서
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.evaluate import evaluate, load_examples

parser = argparse.ArgumentParser()
parser.add_argument("--set", action="append", default=[], help="train 설정 덮어쓰기 (키=값)")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
cfg, model_cfg = params["train"], params["model"]
for kv in args.set:
    key, value = kv.split("=", 1)
    cfg[key] = yaml.safe_load(value)
run_name = cfg["run_name"] or f"r{cfg['lora_r']}-{cfg['target']}-{cfg['quantization']}"
out_dir = Path("outputs") / run_name


def code_version() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "."], capture_output=True, text=True).stdout)
    return {"git_commit": commit, "git_dirty": str(dirty)}


def data_version() -> dict:
    lock = yaml.safe_load(open("dvc.lock", encoding="utf-8"))
    outs = {o["path"]: o["md5"] for stage in lock["stages"].values() for o in stage["outs"]}
    return {"data_prepared_md5": outs["data/prepared"]}


# 1. 모델: QLoRA면 베이스 가중치를 4비트로 불러온다
tok = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg["revision"])
quant = None
if cfg["quantization"] == "4bit":
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], revision=model_cfg["revision"],
                                             dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0})

# 2. LoRA: 어느 층에 어댑터를 붙일지. Qwen3.5는 32층 중 8층만 self_attn이다 (08-3)
targets = {"all-linear": "all-linear",
           "attention": ["q_proj", "k_proj", "v_proj", "o_proj"]}[cfg["target"]]
peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                         target_modules=targets, task_type="CAUSAL_LM",
                         revision=model_cfg["revision"])   # 적지 않으면 adapter_config.json에 null로 남는다 (08-4)

# 3. 학습 설정: 체크포인트는 save_steps마다, 최근 save_total_limit개만 남긴다 (08-4)
sft_args = SFTConfig(
    output_dir=str(out_dir),
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=cfg["grad_accum"],
    learning_rate=cfg["lr"],
    lr_scheduler_type="cosine",
    warmup_steps=0.05,               # 1보다 작으면 전체 단계 대비 비율 (transformers 5에서 warmup_ratio 대체)
    max_length=cfg["max_length"],
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=cfg["save_steps"],
    save_strategy="steps",
    save_steps=cfg["save_steps"],
    save_total_limit=cfg["save_total_limit"],
    per_device_eval_batch_size=8,
    seed=cfg["seed"],
    report_to="mlflow",
    run_name=run_name,
)
train_ds = load_dataset("json", data_files="data/prepared/train.jsonl", split="train")
val_ds = load_dataset("json", data_files="data/prepared/val.jsonl", split="train")
trainer = SFTTrainer(model=model, args=sft_args, train_dataset=train_ds, eval_dataset=val_ds,
                     processing_class=tok, peft_config=peft_config)
trainable, total = trainer.model.get_nb_trainable_parameters()

mlflow.set_experiment("ch08-flower-support")
# 이어서 학습할 때는 같은 MLflow 실행에 이어서 기록한다 (실행 ID를 출력 폴더에 남겨 둔다) (08-4)
run_id_file = out_dir / "mlflow_run_id.txt"
resume_id = run_id_file.read_text().strip() if args.resume and run_id_file.exists() else None
with mlflow.start_run(run_id=resume_id, run_name=None if resume_id else run_name) as run:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id_file.write_text(run.info.run_id)
    if resume_id:
        mlflow.set_tag("resumed", "true")
    mlflow.set_tags(code_version() | data_version() | {
        "stage": "finetune", "base_model": model_cfg["name"], "base_revision": model_cfg["revision"],
        "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "python_version": platform.python_version()})
    mlflow.log_params({f"cfg_{k}": v for k, v in cfg.items()} | {"trainable_params": trainable})
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    train_minutes = (time.time() - start) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # 4. 어댑터만 저장한다 (베이스 모델은 Hub 리비전으로 다시 받는다) (08-4)
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    adapter_mb = sum(f.stat().st_size for f in adapter_dir.glob("*")) / 1e6

    # 5. 테스트 세트 평가 (08-1의 기준선과 같은 함수)
    metrics, results = evaluate(trainer.model, tok, load_examples("data/prepared/test.jsonl"))
    mlflow.log_metrics(metrics | {"train_minutes": train_minutes, "gpu_peak_allocated_gb": peak_gb,
                                  "adapter_mb": adapter_mb})
    mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
    mlflow.log_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), "predictions_test.jsonl")
    print(f"{run_name}: 학습 {train_minutes:.1f}분, GPU 최대 {peak_gb:.1f}GB, 어댑터 {adapter_mb:.1f}MB, "
          f"학습 파라미터 {trainable:,} / {total:,}")
    for k, v in metrics.items():
        print(f"  {k:28s} {v:.4f}")
