# 12장. 통합 MLOps 모니터링

두 프로젝트(A: 꽃 분류, B: 고객 문의 LLM)를 하나의 관측 체계로 묶습니다. Prometheus·Grafana로 서비스·모델·GPU 지표를 모으고, Evidently로 드리프트를 검사하고, Alertmanager로 알림을 보냅니다. 마지막으로 '어두운 사진'이라는 새 촬영 조건을 감지해 데이터를 보강하고, 7장 파이프라인으로 재학습·판정·승인 배포까지 한 바퀴(CT 루프)를 돕니다.

| 폴더·파일 | 내용 | 관련 절 |
|---|---|---|
| `monitoring/compose.yaml` | Prometheus, Grafana, Alertmanager, Pushgateway, node-exporter, 알림 수신기 | 12-1 |
| `monitoring/prometheus/prometheus.yml`, `targets/` | 수집 대상 (GPU 서버는 파일로, 고치면 바로 반영) | 12-1, 12-2 |
| `monitoring/grafana/` | 데이터 소스·대시보드 3개(프로젝트 A, 프로젝트 B, GPU)를 파일로 등록 | 12-1 |
| `monitoring/exporters/compose.yaml` | GPU 서버용: DCGM Exporter(9400), node-exporter(9100) | 12-2 |
| `monitoring/prometheus/rules/alerts.yml` | 알림 규칙 (서비스, 모델, GPU, 드리프트) | 12-4 |
| `monitoring/alertmanager/alertmanager.yml`, `receiver/` | 묶기·반복·억제와 웹훅 수신기 | 12-4 |
| `flowers-api/app/metrics.py`, `features.py` | 추론 서버의 `/metrics`와 예측 기록(입력 특성·예측) | 12-1, 12-3 |
| `flower-support/gateway/metrics.py` | LLM 게이트웨이의 `/metrics`, vLLM 지표 중계(`/vllm-metrics`) | 12-1, 12-2 |
| `flower-support/src/load.py` | 일정한 속도의 LLM 트래픽 흉내 | 12-2, 12-3 |
| `flower-support/monitor/drift_llm.py` | 트레이스 기반 LLM 드리프트 검사 (학습 환경과 분리) | 12-3 |
| `flowers/` | 7장 파이프라인 + 12장 변경 (비공개 저장소로 옮겨 쓰는 폴더) | 12-3, 12-5 |
| `flowers/monitor/drift.py`, `.github/workflows/monitor.yml` | 매시간 드리프트 검사 → Pushgateway·MLflow | 12-3 |
| `flowers/src/reference.py` | 모델 버전마다 기준 데이터(검증 세트 특성·예측)를 만들어 붙인다 | 12-3 |
| `flowers/src/make_night.py`, `simulate_traffic.py` | 새 촬영 조건(어두운 사진) 흉내, 운영 트래픽 흉내 | 12-5 |
| `flowers/src/prepare.py`, `registry.py`, `pipeline/steps.py` | 사본을 원본과 같은 세트로, 조각별 F1, champion 재측정 | 12-5 |

## 빠른 실행

```bash
# 1. 모니터링 스택 (2장 MLflow와 같은 서버)
cd monitoring
echo "GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)" > .env && chmod 600 .env
docker compose up -d
# GPU 서버가 따로 있으면 prometheus/targets/*.yml 의 주소를 그 서버로 바꾸고, GPU 서버에서:
docker compose -f exporters/compose.yaml up -d

# 2. 지표를 내보내는 서비스로 교체
cd ../flowers-api && docker compose up -d --build                       # 6장 서버 + /metrics + 예측 기록
cd ../flower-support && docker compose -f serving/compose.yaml up -d --build   # 10·11장 스택 + /metrics

# 3. 드리프트 검사 (프로젝트 A: 비공개 저장소의 monitor 워크플로, 프로젝트 B: 직접 실행)
cd ../flower-support/monitor && uv run python drift_llm.py --reference-hours 24 --current-hours 6
```

Grafana: http://localhost:3000 (admin / `.env`의 비밀번호), 폴더 **MLOps** 에 대시보드 3개가 있습니다.

## CT 루프 (12-5, 비공개 저장소에서)

```bash
# 0. 지금 champion에 드리프트 기준 데이터 붙이기 (12-3 이전에 만든 모델만, 한 번)
uv run python -m src.reference <champion 버전>

# 1. 데이터를 더하기 전에 지금 세트를 고정 (한 번)
uv run python -m src.freeze_split            # → data/split.csv

# 2. 새 조건 데이터 (실제로는 수집·라벨링) → 기록 → 다시 나누기 → 원격 저장소
uv run python -m src.make_night --fraction 0.25
uv run dvc add data/raw && uv run dvc repro && uv run dvc push
git add data/raw.dvc dvc.lock data/split.csv reports/ && git commit -m "새 조건 데이터" && git push
#    → pipeline 워크플로: 학습 → 조각별 판정(champion도 같은 테스트 세트로 재측정) → challenger

# 3. 사람이 승인: Actions → deploy → Run workflow
```

`dvc repro`는 `data/prepared`를 지우고 다시 만듭니다. 같은 폴더의 사진을 읽는 다른 작업(`simulate_traffic.py` 등)은 그동안 멈추세요.

> **주의.** `flowers/` 폴더는 7장처럼 **여러분의 비공개 저장소** 루트로 옮겨서 씁니다. 학습·배포·드리프트 검사가 여러분 서버의 self-hosted 러너에서 돕니다. 저장소 변수 `PREDICTION_LOG_DIR`에 추론 서버의 `logs` 폴더 경로를 넣으세요.
