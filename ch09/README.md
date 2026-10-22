# 9장. LLM 평가와 프롬프트 관리

8장에서 파인튜닝한 어댑터를 학습 데이터와 다른 분포의 평가 세트로 다시 재고, 요약을 LLM 심판으로 채점하고, 프롬프트를 MLflow 프롬프트 레지스트리로 관리합니다. 약점을 보강해 다시 학습한 모델을 보류 세트·회귀 테스트·요약 심판으로 이루어진 배포 게이트에 통과시켜 레지스트리에 등록합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flower-support/src/make_evalset.py` | 조각(slice)별 평가 세트 합성 (개발용 `evalset`, 보류 `holdout`) | 09-2, 09-5 |
| `flower-support/src/build_evalset.py` | 검토 기록 적용, 학습 데이터와 겹침·다양성 검사 | 09-2 |
| `flower-support/eval/review_*.yaml` | 판단이 갈린 문항의 사람 검토 기록 | 09-2 |
| `flower-support/src/eval_llm.py` | 모델 + 프롬프트 조합 추론과 필드·조각별 채점 | 09-3 ~ 09-5 |
| `flower-support/src/judge.py` | 요약 LLM 심판 (사실성·핵심 포함) + 규칙(한 문장) | 09-3 |
| `flower-support/src/judge_check.py` | 심판 점검: 망가뜨린 요약을 잡는가, 같은 판정을 다시 내는가 | 09-3 |
| `flower-support/src/prompts.py` | 프롬프트 v1·v2를 MLflow 프롬프트 레지스트리에 등록 | 09-4 |
| `flower-support/src/make_augment.py`, `prepare_v2.py` | 약점 보강 데이터 합성, 프롬프트 v2 재학습 데이터 | 09-5 |
| `flower-support/src/gate.py` | 배포 게이트 판정, 어댑터 등록(프롬프트 연결), 승인 시 별칭 이동 | 09-5 |
| `flower-support/src/train.py` 등 | 8장 코드 (학습 데이터 폴더를 설정으로 받도록만 바꿈) | 8장 |

실측에 쓴 합성 데이터(가상의 문의)는 [`dataset/`](dataset/)에 있습니다: 개발용 평가 세트·보류 세트 원본과 보강 데이터. 8장 학습 데이터는 [`../ch08/dataset/`](../ch08/dataset/)의 파일을 씁니다.

## 빠른 실행

```bash
cd flower-support
uv sync --extra cu130
mkdir -p data/raw data/augment eval/raw
cp ../../ch08/dataset/generated.jsonl data/raw/
cp ../dataset/evalset_generated.jsonl ../dataset/holdout_generated.jsonl \
  eval/raw/
cp ../dataset/augment_generated.jsonl data/augment/generated.jsonl
uv run dvc init && uv run dvc add data/raw eval/raw data/augment && uv run dvc repro

export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -m src.prompts register

# 평가 (GPU). 8장 학습 실행 ID로 어댑터를 받아 온다
P=prompts:/flower-support-extract
uv run python -m src.eval_llm --prompt $P/1 --adapter-run <8장 실행 ID> --name r16-v1
# 요약 심판 (API 키가 있는 곳)
export OPENAI_API_KEY=sk-... OPENAI_JUDGE_MODEL=<심판 모델>
uv run python -m src.judge r16-v1

# 재학습 → 보류 세트·회귀 평가 → 게이트
uv run python -m src.train --set data_dir=data/prepared_v2 \
  --set run_name=r16-v2data
uv run python -m src.eval_llm --data holdout --prompt $P/2 --adapter-run <재학습 실행 ID> --name r16v2-v2-holdout
uv run python -m src.eval_llm --data indist --prompt $P/2 --adapter-run <재학습 실행 ID> --name r16v2-v2-indist
uv run python -m src.judge r16v2-v2-holdout
uv run python -m src.gate r16v2-v2 --train-run <재학습 실행 ID> --prompt $P/2 --approve
```
