"""어댑터를 베이스 모델에 병합해 하나의 모델로 저장하고, 병합 전후 출력이 같은지 확인한다 (08-4).

실행:
    uv run python -m src.merge outputs/<실행이름>/adapter
"""

import sys
import time
from pathlib import Path

import torch

from src.evaluate import evaluate, load_examples, load_model

adapter = Path(sys.argv[1])
merged_dir = adapter.parent / "merged"
examples = load_examples("data/prepared/test.jsonl", limit=32)

model, tok = load_model(str(adapter))
before, before_rows = evaluate(model, tok, examples)

start = time.perf_counter()
model = model.merge_and_unload()           # LoRA의 B·A 곱을 원래 가중치에 더하고 어댑터 층을 없앤다
model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print(f"병합·저장 {time.perf_counter() - start:.1f}초 → {merged_dir}")

after, after_rows = evaluate(model, tok, examples)
same = sum(a["output"] == b["output"] for a, b in zip(before_rows, after_rows))
size_gb = sum(f.stat().st_size for f in merged_dir.glob("*")) / 1e9
adapter_mb = sum(f.stat().st_size for f in adapter.glob("*")) / 1e6
print(f"어댑터 {adapter_mb:.1f}MB / 병합 모델 {size_gb:.2f}GB")
print(f"출력이 글자 하나까지 같은 예시: {same}/{len(examples)}")
print(f"전체 필드 정답률: 병합 전 {before['test_all_fields']:.4f} / 병합 후 {after['test_all_fields']:.4f}")
print(f"예시당 시간: 병합 전 {before['test_seconds_per_example']:.3f}초 / 병합 후 {after['test_seconds_per_example']:.3f}초")
del model
torch.cuda.empty_cache()
