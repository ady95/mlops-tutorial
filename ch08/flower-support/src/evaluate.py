"""테스트 세트로 JSON 추출 성능 평가 (08-1 기준선, 08-5 파인튜닝 결과).

실행:
    uv run python -m src.evaluate                          # 베이스 모델 (파인튜닝 전 기준선)
    uv run python -m src.evaluate --adapter outputs/<실행이름>/adapter
"""

import argparse
import json
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schema import parse_output, score


def load_examples(path: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


@torch.no_grad()
def evaluate(model, tok, examples: list[dict], batch_size: int = 16) -> tuple[dict, list[dict]]:
    """예시마다 JSON을 생성해 채점한다. (지표, 예시별 결과)"""
    tok.padding_side = "left"           # 생성할 때는 왼쪽을 채워야 문장 끝이 나란해진다
    model.eval()
    results, start, new_tokens = [], time.perf_counter(), 0
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        texts = [tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True,
                                         **ex["chat_template_kwargs"]) for ex in batch]
        inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        for ex, ids in zip(batch, out[:, inputs["input_ids"].shape[1]:]):
            text = tok.decode(ids, skip_special_tokens=True)
            new_tokens += int((ids != tok.pad_token_id).sum())
            gold = json.loads(ex["completion"][0]["content"])
            pred = parse_output(text)
            results.append({"id": ex["id"], "output": text, **score(pred, gold)})
    elapsed = time.perf_counter() - start
    n = len(results)
    keys = ["json_valid", "schema_valid", "category", "urgency", "sentiment", "order_id", "product"]
    metrics = {f"test_{k}": sum(r[k] for r in results) / n for k in keys}
    metrics["test_all_fields"] = sum(all(r[k] for k in keys[2:]) for r in results) / n
    metrics["test_seconds_per_example"] = elapsed / n
    metrics["test_tokens_per_second"] = new_tokens / elapsed
    return metrics, results


def load_model(adapter: str | None = None, load_4bit: bool = False):
    cfg = yaml.safe_load(open("params.yaml", encoding="utf-8"))["model"]
    tok = AutoTokenizer.from_pretrained(cfg["name"], revision=cfg["revision"])
    kwargs = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg["name"], revision=cfg["revision"], dtype=torch.bfloat16,
                                                 device_map={"": 0}, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


if __name__ == "__main__":
    import mlflow

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model, tok = load_model(args.adapter)
    examples = load_examples(f"data/prepared/{args.split}.jsonl", args.limit)
    metrics, results = evaluate(model, tok, examples, args.batch_size)
    for k, v in metrics.items():
        print(f"{k:28s} {v:.4f}")

    mlflow.set_experiment("ch08-flower-support")
    with mlflow.start_run(run_name="baseline" if not args.adapter else f"eval-{args.adapter.split('/')[-2]}"):
        mlflow.set_tags({"stage": "baseline" if not args.adapter else "evaluation",
                         "adapter": args.adapter or "none", "split": args.split})
        mlflow.log_params(yaml.safe_load(open("params.yaml", encoding="utf-8"))["model"])
        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), f"predictions_{args.split}.jsonl")
