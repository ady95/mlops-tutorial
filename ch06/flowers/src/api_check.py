"""테스트 세트의 원본 사진을 API로 보내 정확도를 확인한다 (06-1, 06-5).

실제 사용자는 크기가 제각각인 원본 사진을 보낸다. 그래서 전처리를 마친 data/prepared가 아니라
같은 이름의 원본(data/raw)을 보낸다. 정확도가 5장 오프라인 평가와 다르면 전처리가 어긋난 것이다.

실행:
    uv run python src/api_check.py http://localhost:8000
"""

import sys
from collections import Counter
from pathlib import Path

import httpx

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
files = sorted(Path("data/prepared/test").glob("*/*.jpg"))

correct, versions, latencies = 0, Counter(), []
with httpx.Client(base_url=url, timeout=30) as client:
    for path in files:
        raw = Path("data/raw") / path.parent.name / path.name
        r = client.post("/predict", files={"file": (raw.name, raw.read_bytes(), "image/jpeg")})
        r.raise_for_status()
        body = r.json()
        correct += body["class"] == path.parent.name
        versions[body["model_version"]] += 1
        latencies.append(body["latency_ms"])

latencies.sort()
print(f"{len(files)}장 중 {correct}장 정답, 정확도 {correct / len(files):.4f}")
print(f"응답한 모델 버전: {dict(versions)}")
print(f"서버 처리 시간 중앙값 {latencies[len(latencies) // 2]:.1f}ms")
