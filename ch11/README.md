# 11장. LLM 서비스 모니터링

10장의 게이트웨이가 요청마다 MLflow 트레이스를 남기게 하고, 정답 없이 볼 수 있는 신호·표본 심판·상담원 피드백으로 품질을 지켜봅니다. 새 상품이 생긴 상황에서 알림을 받아 프롬프트를 고치고 다시 배포하는 개선 루프를 돕니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flower-support/gateway/app.py` | 요청별 트레이스(버전·토큰·신호), `request_id`, `/feedback`, 모델·프롬프트 짝 검사 | 11-1 ~ 11-5 |
| `flower-support/gateway/guards.py` | 민감정보 가리기, 근거·상품 목록 검사, 지시 주입 의심, 분류 확신도 | 11-4 |
| `flower-support/src/simulate.py` | 운영 트래픽 흉내 (평소 구간 → 새 상품 문의 구간, 일부 연락처 포함) | 11-3, 11-4 |
| `flower-support/src/monitor.py` | 구간별 사용량·신호 집계, 표본 요약 심판(트레이스에 피드백), 알림 | 11-1, 11-3, 11-4 |
| `flower-support/src/make_drift.py` | 새 상품("선인장 화분", "프리저브드 장미 박스") 문의 합성 | 11-4 |
| `flower-support/src/collect_feedback.py` | 상담원이 틀렸다고 한 요청을 평가 세트 형식으로 내보내기 | 11-5 |
| `flower-support/src/schema.py`, `prompts.py` | 판매 목록(`CATALOG`)과 프롬프트 v3 | 11-5 |
| `flower-support/src/` 그 밖 | 8~10장 코드 | 8~10장 |

실측에 쓴 새 상품 문의 80건(합성, 가상의 문의)은 [`dataset/drift_generated.jsonl`](dataset/)에 있습니다. `flower-support/eval/raw/`에 복사한 뒤 **python src/make_drift.py --convert-only** 로 평가 형식(`eval/drift.jsonl`)을 만들거나, API로 직접 생성하세요.

## 빠른 실행

```bash
cd flower-support
# 10장 스택에서 게이트웨이만 바뀝니다
docker compose -f serving/compose.yaml up -d --build
export MLFLOW_TRACKING_URI=http://localhost:5000

# 트래픽과 모니터링
uv run python -m src.simulate --normal 300 --drift 160
export OPENAI_API_KEY=sk-... OPENAI_JUDGE_MODEL=<심판 모델>
uv run python -m src.monitor --phases eval/outputs/simulate_phases.json --sample 40

# 개선 루프: 프롬프트 v3 → 평가 → 게이트 → 승인 → 반영
uv run python -m src.prompts register 3
uv run python -m src.eval_served --model flower-support-adapter-v1 --prompt prompts:/flower-support-extract/3 --data eval/holdout.jsonl --name v1p3-holdout
uv run python -m src.eval_served --model flower-support-adapter-v1 --prompt prompts:/flower-support-extract/3 --data data/prepared/test.jsonl --name v1p3-indist
uv run python -m src.judge v1p3-holdout
uv run python -m src.gate v1p3 --train-run <v1 학습 실행 ID> --prompt prompts:/flower-support-extract/3
uv run python -m src.promote approve && curl -s -X POST localhost:8002/reload -H "X-Reload-Token: $RELOAD_TOKEN"

# 상담원 피드백 회수
uv run python -m src.collect_feedback --days 7
```
