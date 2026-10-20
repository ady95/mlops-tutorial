# 13장. AI 에이전트를 활용한 MLOps 자동화

표준 OpenAI API로 작은 운영 보조 에이전트를 만들고, 파이프라인 코드 수정·장애 분석·실험 해석·재학습 추천을 맡겨 정답과 비교합니다. 에이전트는 읽기 도구만 쓰고, 무언가를 바꾸는 일은 GitHub 이슈로 승인을 요청하는 것뿐입니다. 사람이 `/approve`를 달면 비공개 저장소의 워크플로가 허용된 조치만 실행합니다.

| 폴더·파일 | 내용 | 관련 절 |
|---|---|---|
| `ops-agent/agent/core.py` | 에이전트의 반복(도구 호출 → 결과 → 다음 판단), 도구 정의, MLflow 트레이스 | 13-1 |
| `ops-agent/agent/sanitize.py` | 도구 결과에서 내부 주소·계정 경로·키·연락처를 지운다 | 13-2 |
| `ops-agent/agent/tools_code.py` | 샌드박스 안의 파일 읽기·쓰기·부분 바꾸기·테스트 실행 | 13-1 |
| `ops-agent/agent/tools_ops.py` | Prometheus 조회, 알림 이력, 수집 상태, 정해 둔 서비스의 로그 | 13-2 |
| `ops-agent/agent/tools_mlflow.py`, `grounding.py` | MLflow 실행·레지스트리 조회, 답의 숫자가 도구 결과에 있었는지 검사 | 13-3 |
| `ops-agent/agent/tools_retrain.py` | 모델별 기준 데이터 요약, 운영 구간 요약, 파이프라인 설정 읽기 | 13-4 |
| `ops-agent/agent/tools_action.py` | 조치 제안(승인 요청 이슈 열기) — 에이전트의 유일한 '바꾸는' 도구 | 13-5 |
| `ops-agent/codegen.py`, `cases/codegen/` | 코딩 에이전트 실행과 숨은 인수 테스트 | 13-1 |
| `ops-agent/triage.py`, `cases/incidents.yaml` | 장애 분석, 12장의 실제 장애 4건과 채점 기준 | 13-2 |
| `ops-agent/analyze.py` | 실험 기록 분석 | 13-3 |
| `ops-agent/advise.py` | 재학습 후보 추천 (MLflow `retrain-recommendations`에 근거 기록) | 13-4 |
| `ops-agent/respond.py` | 조사 후 필요하면 승인 요청을 여는 에이전트 | 13-5 |
| `flowers/src/actions.py`, `.github/workflows/approve.yml` | 승인 실행기: 이슈의 JSON을 검사하고 허용된 조치(deploy, restore, silence, retrain)만 실행 | 13-5 |

## 빠른 실행

```bash
cd ops-agent
uv sync
cp services.example.yaml services.yaml        # 로그를 읽을 서비스와 명령
export OPENAI_API_KEY=sk-... OPENAI_MODEL=<모델>
export MLFLOW_TRACKING_URI=http://localhost:5000 PROMETHEUS_URL=http://localhost:9090

uv run python triage.py --cases cases/incidents.yaml --repeat 3                 # 13-2
uv run python analyze.py --question "ch04-flowers에서 배포 후보를 골라 줘" --repeat 3   # 13-3
PIPELINE_DIR=../flowers uv run python advise.py --window-start <시작> --window-end <끝> --with-production-summary   # 13-4

# 13-5: 이슈 쓰기 권한만 있는 토큰 (fine-grained PAT, Issues: Read and write)
export GITHUB_TOKEN=... GITHUB_REPO=<계정>/mlops-pipeline
uv run python respond.py --event "FlowersLowConfidence 발생" --start <시작> --end <끝>

# 13-1: 코딩 에이전트 (테스트용 가상환경은 테스트 프로세스에만 넘긴다)
TEST_VENV=$PWD/.venv-flowers uv run python codegen.py --source ../flowers --tasks cases/codegen/tasks.yaml --tools edit --repeat 3
```

> **주의.** `flowers/` 폴더는 7장처럼 **여러분의 비공개 저장소** 루트로 옮겨서 씁니다. `approve` 워크플로는 저장소 권한(OWNER·MEMBER·COLLABORATOR)이 있는 사람의 `/approve` 댓글에만 반응하고, `agent-proposal` 라벨이 붙은 이슈만 봅니다. 라벨은 미리 만들어 두세요.
