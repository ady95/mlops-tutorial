"""8장 어댑터를 vLLM이 알아보는 이름으로 바꾼다 (10-1).

학습은 텍스트 전용 클래스(Qwen3_5ForCausalLM)로 해서 어댑터의 층 이름이 `model.layers.N...`이다.
vLLM은 Qwen3.5를 이미지 입력까지 받는 구조(Qwen3_5ForConditionalGeneration)로 올리고, 이 구조에서 언어 모델 층은
`model.language_model.layers.N...`이다. 이름이 맞지 않으면 vLLM은 오류 없이 어댑터를 무시한다.

safetensors 파일은 [헤더 길이 8바이트][JSON 헤더][텐서 데이터]로 되어 있고, 텐서 이름은 헤더에만 있다.
그래서 헤더의 이름만 바꿔 쓰면 되고, torch 없이 빠르게 동작한다 (게이트웨이 컨테이너에서도 쓴다).

실행:
    python src/adapter_for_vllm.py <원본 어댑터 폴더> <저장할 폴더>
"""

import json
import shutil
import struct
import sys
from pathlib import Path

OLD, NEW = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."


def convert(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    with open(src / "adapter_model.safetensors", "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        data = f.read()
    renamed = {k.replace(OLD, NEW): v for k, v in header.items()}
    tensors = [k for k in renamed if k != "__metadata__"]
    assert all(".language_model.layers." in k for k in tensors), "예상한 이름 형식이 아닙니다"
    raw = json.dumps(renamed, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)            # 헤더 길이를 8바이트 단위로 맞춘다 (safetensors 규칙)
    with open(dst / "adapter_model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + data)
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    return len(tensors)


if __name__ == "__main__":
    n = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{n}개 텐서 이름 변경 → {sys.argv[2]}")
