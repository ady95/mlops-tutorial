# 4장. 모델 학습과 실험 관리

3장의 꽃 데이터(DVC 파이프라인)로 사전학습 모델을 전이학습하고, 모든 실험을 MLflow에 기록·비교·재현합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flowers/src/download.py`, `validate.py`, `prepare.py`, `dvc.yaml` | 3장과 같은 데이터 파이프라인 | 3장 |
| `flowers/src/train.py` | 전이학습 + MLflow 기록 (코드·데이터 버전, 혼동 행렬, CPU 지연시간, 모델) | 04-1 ~ 04-3 |
| `flowers/src/best_run.py` | MLflow 검색 API로 실행 비교 | 04-4 |
| `flowers/src/reproduce.py` | 기록된 실행을 같은 코드·데이터·설정으로 재학습 | 04-5 |

## 빠른 실행

```bash
# 2장의 MLflow 서버가 실행 중이어야 합니다
cd flowers
uv sync --extra cpu                      # NVIDIA GPU: uv sync --extra cu130
uv run python src/download.py
uv run dvc init --subdir && uv run dvc repro   # 데이터 준비 (DVC 원격 설정은 3장 참고)
git add -A && git commit -m "ch04: 데이터 준비"   # 실행에 Git 커밋이 기록됩니다
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python src/train.py --model mobilenet_v3_small --freeze-backbone --epochs 5
uv run python src/best_run.py
```
