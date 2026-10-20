# 14장. 실전 통합 MLOps 플랫폼 완성

세 프로젝트(A 꽃 분류, B 고객 문의 LLM, C 번호판 인식)가 함께 쓰는 MLflow 서버를 계보 도구로 점검하고, 검출·인식 두 모델로 된 번호판 인식 서비스로 여러 모델의 판정·배포·롤백을 다룹니다. 추론 서버를 Kubernetes(kind)에 올리고, 켜져 있는 시스템 전체를 한 번에 점검합니다.

| 폴더·파일 | 내용 | 관련 절 |
|---|---|---|
| `platform/lineage.py`, `lineage.yaml` | 운영 모델의 계보 조회(`show`)·점검(`audit`), 실험 프로젝트 태그, 은퇴(`retire`) | 14-1 |
| `plates/src/synth.py` | 합성 번호판 장면 (글꼴로 그린 번호판 + 무작위 배경) | 14-2 |
| `plates/src/models.py` | 검출(MobileNetV3-Small 좌표 회귀)·인식(CRNN + CTC) 모델, 두 모델 사이의 약속(자르기·전처리) | 14-2 |
| `plates/src/train.py` | 모델 하나씩 학습·등록. 인식 모델은 정답 상자 조각(`--crops gt`) 또는 검출 모델이 자른 조각(`--crops detector`) | 14-2 |
| `plates/src/evaluate.py` | 검출만·인식만·끝단(end-to-end) 평가 | 14-2 |
| `plates/src/registry.py` | 짝 모델의 champion과 이어서 판정, 통과한 조합 목록 | 14-2 |
| `plates/src/deploy.py` | 모델 하나씩 배포·롤백. 바꾼 뒤 조합이 판정을 통과한 조합인지 확인 | 14-2, 14-3 |
| `plates/app/` | plate-api: 두 모델을 각자의 champion으로 서빙, `/reload?model=` | 14-2 |
| `flowers/` | 비공개 파이프라인 저장소의 사본 (등록 태그에 공통 계보 키 추가) | 14-3 |
| `flowers-api/` | 꽃 분류 추론 서버, `MODEL_VERSION`으로 버전 고정 (Kubernetes용) | 14-4 |
| `k8s/` | kind 클러스터 설정, Deployment·Service, HPA, 부하 도구 | 14-4 |
| `platform/validate.py`, `validate.yaml` | 최종 시스템 점검 (서비스·서빙 버전·롤백·드리프트·알림·계보) | 14-5 |

## 빠른 실행

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000

# 14-1 계보
cd platform && uv sync
uv run python lineage.py audit
uv run python lineage.py show flowers-classifier@champion

# 14-2 번호판 (한글 글꼴: sudo apt install fonts-noto-cjk)
cd ../plates && uv sync --extra cpu
uv run python -m src.synth --font /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc
uv run python -m src.train detector --epochs 8
uv run python -m src.train recognizer --epochs 20 --crops gt
uv run python -m src.registry gate detector 1 --partner 1 && uv run python -m src.deploy deploy detector --no-reload
uv run python -m src.registry gate recognizer 1 && uv run python -m src.deploy deploy recognizer --no-reload
echo "RELOAD_TOKEN=$(openssl rand -hex 16)" > .env && docker compose up -d --build     # :8010
uv run python -m src.train recognizer --epochs 20 --crops detector --detector-version 1
uv run python -m src.registry gate recognizer 2
set -a && . ./.env && set +a && uv run python -m src.deploy deploy recognizer

# 14-4 Kubernetes (kind, kubectl, envsubst 필요)
cd ../flowers-api && uv lock && docker compose build
cd ../k8s && kind create cluster --name mlops --config kind-config.yaml
kind load docker-image flowers-api:0.4.0 --name mlops
kubectl create namespace mlops
kubectl -n mlops create secret generic flowers-api --from-literal=reload-token=$(openssl rand -hex 16)
HOST_IP=$(hostname -I | awk '{print $1}') envsubst '$HOST_IP' < flowers-api.yaml | kubectl apply -f -

# 14-5 점검
cd ../platform && uv run python validate.py
```

번호판 데이터는 합성이라 저장소에 넣지 않습니다(`data/`는 `.gitignore`). 같은 시드면 같은 장면이 만들어집니다.
