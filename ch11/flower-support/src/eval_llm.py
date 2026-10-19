"""평가 세트로 모델 + 프롬프트 조합을 평가한다: 추론과 필드 채점 (09-3, 09-4, 09-5).

요약 품질은 API로 심판하므로 src/judge.py가 따로 맡는다 (GPU 서버와 API 키를 나눠 둘 수 있게).

실행:
    uv run python -m src.eval_llm --prompt prompts:/flower-support-extract/1                 # 베이스
    uv run python -m src.eval_llm --prompt prompts:/flower-support-extract/1 --adapter-run <MLflow 실행 ID>
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import messages
from src.schema import parse_output, score

KEYS = ["category", "urgency", "sentiment", "order_id", "product"]
SLICE_KEYS = {"길고 산만한 문의": "long", "여러 요청": "multi", "경계 사례": "boundary",
              "짧은 문의": "short", "주문번호 함정": "order_trap", "지시 주입": "injection", "학습 분포": "indist"}

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True, help="prompts:/이름/버전 또는 prompts:/이름@별칭")
parser.add_argument("--adapter-run", help="어댑터를 기록한 8장 학습 실행의 MLflow ID (없으면 베이스 모델)")
parser.add_argument("--name", required=True, help="이 평가의 이름 (예: r16-v1)")
parser.add_argument("--data", choices=["evalset", "holdout", "indist"], default="evalset",
                    help="evalset = 개발용 평가 세트, holdout = 보류 세트(09-5 최종 판정), indist = 8장 테스트 세트(회귀 확인)")
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()

model_cfg = yaml.safe_load(open("params.yaml", encoding="utf-8"))["model"]
prompt = mlflow.genai.load_prompt(args.prompt)
tok = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg["revision"])
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], revision=model_cfg["revision"],
                                             dtype=torch.bfloat16, device_map={"": 0})
if args.adapter_run:
    from peft import PeftModel
    adapter_dir = mlflow.artifacts.download_artifacts(run_id=args.adapter_run, artifact_path="adapter")
    model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

if args.data != "indist":
    rows = [json.loads(line) for line in open(f"eval/{args.data}.jsonl", encoding="utf-8")]
else:  # 8장 형식(prompt/completion)에서 문의와 정답만 꺼낸다 — 지시문은 --prompt로 바꿔 끼운다
    rows = [{"id": ex["id"], "slice": "학습 분포", "inquiry": ex["prompt"][1]["content"],
             "gold": json.loads(ex["completion"][0]["content"])}
            for ex in map(json.loads, open("data/prepared/test.jsonl", encoding="utf-8"))]
results, start = [], time.perf_counter()
with torch.no_grad():
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [tok.apply_chat_template(messages(prompt, r["inquiry"]), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for r in batch]
        inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        for r, ids in zip(batch, out[:, inputs["input_ids"].shape[1]:]):
            text = tok.decode(ids, skip_special_tokens=True)
            pred = parse_output(text)
            s = score(pred, r["gold"] | {"summary": None})
            results.append({"id": r["id"], "slice": r["slice"], "inquiry": r["inquiry"], "output": text,
                            "summary": (pred or {}).get("summary"), **s,
                            "all_fields": all(s[k] for k in KEYS)})
elapsed = time.perf_counter() - start

prefix = "indist" if args.data == "indist" else "eval"
metrics = {f"{prefix}_{k}": sum(r[k] for r in results) / len(results)
           for k in ["json_valid", "schema_valid", *KEYS, "all_fields"]}
by_slice = defaultdict(list)
for r in results:
    by_slice[r["slice"]].append(r["all_fields"])
slice_metrics = {s: sum(v) / len(v) for s, v in by_slice.items()}
metrics[f"{prefix}_seconds_per_example"] = elapsed / len(results)

out_path = Path("eval/outputs") / f"{args.name}.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

mlflow.set_experiment("ch09-evaluation")
with mlflow.start_run(run_name=args.name) as run:
    mlflow.set_tags({"prompt_uri": args.prompt, "prompt_version": str(prompt.version),
                     "adapter_run": args.adapter_run or "none", "base_revision": model_cfg["revision"],
                     "data": args.data})
    mlflow.log_metrics(metrics | ({f"slice_{SLICE_KEYS[s]}": v for s, v in slice_metrics.items()}
                                  if args.data != "indist" else {}))
    mlflow.log_dict(slice_metrics, "slice_all_fields.json")
    mlflow.log_artifact(str(out_path))
    Path(out_path.with_suffix(".run_id")).write_text(run.info.run_id)

print(f"{args.name}: 프롬프트 v{prompt.version}, 어댑터 {args.adapter_run or '없음'}")
for k, v in metrics.items():
    print(f"  {k:26s} {v:.4f}")
for s, v in slice_metrics.items():
    print(f"  [{s}] 다섯 필드 정답 {v:.2f}")
