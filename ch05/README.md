# 5장. 모델 버전 관리와 평가

4장에서 학습한 모델을 MLflow 모델 레지스트리에 등록하고, 테스트 세트로 평가한 뒤, 배포 승인 기준을 통과한 버전만 `champion` 별칭으로 승격합니다. 문제가 생기면 별칭만 바꿔 이전 버전으로 롤백합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flowers/src/register.py` | 학습 실행의 모델을 레지스트리 버전으로 등록 (코드·데이터 버전 태그) | 05-2 |
| `flowers/src/evaluate.py` | 테스트 세트 평가 (`mlflow.models.evaluate`, macro F1, CPU 지연시간) | 05-3 |
| `flowers/src/promote.py` | 배포 승인 기준(`params.yaml`의 `gate`) 판정, challenger/champion 별칭 | 05-4 |
| `flowers/src/rollback.py` | champion을 previous 버전으로 되돌리기 | 05-5 |
| `flowers/src/predict_champion.py` | 서비스 쪽 예: `models:/flowers-classifier@champion` 으로 불러와 예측 | 05-1, 05-5 |

## 빠른 실행

```bash
# 2장의 MLflow 서버, 4장의 학습 실행(run_id)이 필요합니다
cd flowers
uv sync --extra cpu
# 데이터 준비는 3·4장과 같습니다 (download.py → dvc init --subdir → dvc repro → git commit)
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python src/register.py <run_id>      # → version N
uv run python src/evaluate.py N
uv run python src/promote.py N              # 판정, 통과하면 challenger
uv run python src/promote.py N --approve    # 사람이 승인 → champion
uv run python src/rollback.py "사유"         # 문제 시 previous로 롤백
```
