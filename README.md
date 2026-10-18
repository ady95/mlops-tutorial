# MLOps 따라하기 — 예제 코드

위키독스 전자책 **《MLOps 따라하기 — 딥러닝부터 LLM까지, AI 모델 학습·배포·운영 자동화》** 의 실습 예제 코드 저장소입니다.

- 책 보기: https://wikidocs.net/book/21726

## 이 저장소에서 만드는 것

로컬(단일 서버 + Docker)에서 실행되는 MLflow 중심의 통합 AI 모델 관리·운영 플랫폼을 단계적으로 구축합니다.

| 프로젝트 | 내용 | 관련 장 |
|---|---|---|
| A. 딥러닝 모델 MLOps | PyTorch 이미지 분류 모델 학습 → MLflow 기록 → Docker API 배포 → 모니터링 | 3~7장 |
| B. 로컬 LLM LLMOps | 소형 오픈소스 LLM LoRA 파인튜닝 → 평가·프롬프트 관리 → vLLM 서빙 | 8~11장 |
| C. 실전 MLOps 플랫폼 | 다중 모델 관리, 신규 데이터 기반 재학습·성능 비교·롤백 | 12~14장 |

## 구성

장별 디렉터리(`ch01/` ~ `ch14/`)로 제공하며, 책 집필에 맞춰 순차적으로 추가됩니다.

## 핵심 스택

MLflow · DVC · ZenML · PyTorch · Hugging Face (TRL / PEFT) · vLLM · FastAPI · Docker · GitHub Actions · Prometheus / Grafana

각 도구의 기준 버전은 책 부록의 버전 대장을 참고하세요.

## 실습 환경

| 환경 | 실습 범위 |
|---|---|
| CPU 노트북 (RAM 16GB 이상) | 1~7장 (데이터 관리, 학습, 배포, 파이프라인) |
| GPU 24GB급 (RTX 3090 등) | 8~12장 (LoRA·QLoRA, 로컬 LLM 서빙) |
| Kubernetes GPU 노드 | 14장 확장 실습 |

## 라이선스

[MIT License](LICENSE)
