"""트래픽 흉내에 쓰는 문의 목록 (11장 simulate.py, 12장 load.py가 함께 쓴다).

평소 문의는 8장 테스트 세트와 9장 평가 세트에서, 새 상품 문의는 make_drift.py에서 가져온다.
"""

import json
import random

normal = [json.loads(line)["prompt"][1]["content"] for line in open("data/prepared/test.jsonl", encoding="utf-8")]
normal += [json.loads(line)["inquiry"] for line in open("eval/evalset.jsonl", encoding="utf-8")]
drift = [json.loads(line)["inquiry"] for line in open("eval/raw/drift_generated.jsonl", encoding="utf-8")]
CHANNELS = ["web", "web", "app", "kakao"]          # 문의가 들어온 경로: 사용량 집계용


def with_contact(text: str, rng: random.Random) -> str:
    """10% 정도의 고객은 연락처를 함께 남긴다 (가짜 번호)."""
    r = rng.random()
    if r < 0.06:
        return text + f" 연락은 010-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}로 부탁드려요."
    if r < 0.10:
        return text + f" 답장은 user{rng.randint(1, 999)}@example.com 으로 주세요."
    return text
