"""병합 모델을 4비트 가중치(W4A16, GPTQ)로 양자화한다 (10-3).

llm-compressor는 학습 환경과 다른 torch·transformers 버전을 요구하므로 이 폴더의 별도 환경에서 실행한다.

실행 (quantize 폴더에서):
    uv run python quantize.py <병합 모델 폴더> <저장 폴더> <보정용 학습 데이터 jsonl>
"""

import json
import sys
import time

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

src, dst, calib_path = sys.argv[1], sys.argv[2], sys.argv[3]
N_CALIB = 256          # 보정(calibration)에 쓸 예시 수: 실제 입력과 비슷한 글로 각 층의 출력 범위를 잰다

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype="auto")
rows = [json.loads(line) for line in open(calib_path, encoding="utf-8")][:N_CALIB]
texts = [tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False, enable_thinking=False) for r in rows]
calib = Dataset.from_list([dict(tok(t, truncation=True, max_length=1024)) for t in texts])

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
start = time.perf_counter()
oneshot(model=model, dataset=calib, recipe=recipe, max_seq_length=1024, num_calibration_samples=len(calib))
print(f"양자화 {time.perf_counter() - start:.0f}초")
model.save_pretrained(dst, save_compressed=True)
tok.save_pretrained(dst)
print(f"저장 → {dst}")
