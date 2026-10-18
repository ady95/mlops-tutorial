# 8장. LLM 모델 학습 및 파인튜닝 관리

프로젝트 B: 꽃 배달 쇼핑몰의 고객 문의를 JSON(유형·긴급도·감정·주문번호·상품·요약)으로 정리하도록 Qwen3.5-4B를 LoRA·QLoRA로 파인튜닝하고, 실험을 MLflow로 비교합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flower-support/src/schema.py` | 라벨 목록, 판단 기준, 시스템 프롬프트, 출력 파싱·채점 | 08-1 |
| `flower-support/src/make_dataset.py` | LLM API로 학습 데이터 합성 (라벨 먼저 → 문의 작성 → 다른 모델로 검수) | 08-1 |
| `flower-support/src/prepare.py` | 품질 검사, 유형별 층화 분할, TRL 학습 형식 변환 | 08-1 |
| `flower-support/src/evaluate.py` | 테스트 세트 채점 (기준선과 파인튜닝 결과) | 08-1, 08-5 |
| `flower-support/src/train.py` | LoRA·QLoRA 학습, 체크포인트, MLflow 기록, 이어서 학습 | 08-3 ~ 08-5 |
| `flower-support/src/merge.py` | 어댑터를 베이스에 병합하고 병합 전후 출력 비교 | 08-4 |
| `flower-support/src/compare.py` | MLflow 실험 비교표 | 08-5 |
| `flower-support/params.yaml`, `dvc.yaml` | 모델 리비전, 학습 설정, 데이터 파이프라인 | 08-2, 08-3 |

학습 데이터(`data/raw/generated.jsonl`, 합성 문의 3,000건)는 저장소에 넣지 않았습니다. `make_dataset.py`로 직접 만들거나, 같은 방식으로 여러분의 과제에 맞게 바꿔 쓰세요. 생성 결과는 사용한 모델에 따라 달라집니다.

## 빠른 실행

```bash
# GPU 서버 (CUDA 13 드라이버). 2장 MLflow 서버가 실행 중이어야 합니다
cd flower-support
uv sync --extra cu130

# 1. 데이터 합성 (OpenAI API 또는 호환 서버)
export OPENAI_API_KEY=sk-... OPENAI_MODEL=<모델> OPENAI_CHECK_MODEL=<검수 모델>
uv run python src/make_dataset.py
uv run dvc init && uv run dvc add data/raw && uv run dvc repro

# 2. 모델 받기 (리비전 고정)
uv run hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

# 3. 기준선 → 파인튜닝 → 비교
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -m src.evaluate                           # 파인튜닝 전
uv run python -m src.train                              # LoRA (GPU 24GB 권장)
uv run python -m src.train --set quantization=4bit      # QLoRA (GPU 12GB)
uv run python -m src.compare
```

학습·평가 스크립트는 Git 커밋과 DVC 데이터 해시를 MLflow 태그로 남기므로, `git init` 후 커밋한 상태에서 실행하세요.
