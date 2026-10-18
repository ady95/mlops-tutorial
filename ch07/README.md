# 7장. 파이프라인 자동화와 CI/CD

3~6장에서 손으로 하던 일(데이터 준비 → 학습 → 등록 → 평가 → 승인 판정 → 배포)을 ZenML 파이프라인으로 묶고, GitHub Actions로 자동 실행합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `zenml-server/compose.yaml` | ZenML 서버 (Docker) | 07-1 |
| `flowers/src/training.py` | 4장 학습 스크립트를 함수로 정리 (`train(config, experiment)`) | 07-2 |
| `flowers/src/registry.py` | 등록·평가·승인 판정(`judge`) — 5장 로직을 함수로 정리 | 07-2 |
| `flowers/pipeline/steps.py` | 파이프라인 단계 5개 | 07-2 |
| `flowers/pipeline/run.py` | 파이프라인 정의와 실행 진입점 | 07-3 |
| `flowers/tests/` | 승인 판정과 학습 유틸의 단위 테스트 | 07-4 |
| `flowers/.github/workflows/ci.yml` | 모든 push·PR에서 단위 테스트 | 07-4 |
| `flowers/.github/workflows/pipeline.yml` | 학습 관련 파일이 바뀌면 테스트 → 학습 파이프라인 | 07-4 |
| `flowers/.github/workflows/deploy.yml` | 수동 실행: challenger → champion 승격, 추론 서버 reload | 07-5 |
| `flowers/src/deploy.py` | 승격·reload·버전 확인, 실패하면 레지스트리를 되돌림 | 07-5 |
| `flowers/src/download.py` 등 | 3장 데이터 파이프라인 (그대로) | 3장 |

> **주의.** 이 폴더 안의 `.github/workflows/`는 이 저장소에서는 동작하지 않습니다(GitHub는 저장소 루트의 워크플로만 읽습니다). `flowers/` 폴더를 **여러분의 비공개 저장소** 루트로 옮겨서 쓰세요. 학습·배포 작업은 여러분 서버의 self-hosted 러너에서 돌기 때문에, 공개 저장소에 붙이면 남이 보낸 PR이 여러분 서버에서 실행될 수 있습니다.

## 빠른 실행

### 1. ZenML 서버

```bash
cd zenml-server
cp .env.example .env          # ZENML_ADMIN_PASSWORD를 바꾸세요
docker compose up -d
# http://localhost:8080 에서 관리자 계정으로 로그인
```

### 2. 파이프라인을 로컬에서 한 번 실행

```bash
# 2장 MLflow 서버와 3장의 dvc 원격 저장소 데이터가 있어야 합니다
# flowers/를 새 폴더(나중에 비공개 저장소가 될 곳)로 복사한 뒤
git init && uv sync --extra cpu
uv run dvc init
uv run dvc remote add -d rustfs s3://dvc/flowers
uv run dvc remote modify rustfs endpointurl http://localhost:9000
uv run dvc remote modify --local rustfs access_key_id mlops-admin
uv run dvc remote modify --local rustfs secret_access_key mlops-secret-key
# 3장에서 만든 data/raw.dvc 와 dvc.lock 을 복사해 오면 dvc pull로 데이터를 받습니다
git add -A && git commit -m "첫 커밋"

export MLFLOW_TRACKING_URI=http://localhost:5000
export ZENML_STORE_URL=http://localhost:8080
export ZENML_STORE_USERNAME=admin ZENML_STORE_PASSWORD=change-this-password
uv run python -m pipeline.run
cat gate_result.json
```

`git_dirty=True`로 기록되면 승인 판정에서 불가가 됩니다. 커밋하지 않은 파일이 있는지 `git status`로 확인하세요.

### 3. CI/CD (비공개 저장소 + self-hosted 러너)

```bash
# 러너용 서비스 계정과 API 키 (키는 화면에 찍지 말고 파일로)
uv run zenml service-account create ci-runner
uv run zenml service-account api-key ci-runner create ci-runner-key > ~/ci-runner-key.out 2>&1
chmod 600 ~/ci-runner-key.out  # 저장소 밖에 두고, 시크릿 등록 후 지웁니다
```

1. GitHub에 비공개 저장소를 만들고 push합니다.
2. **Settings → Actions → Runners → New self-hosted runner** 의 안내대로 러너를 설치합니다. 라벨에 `mlops-cpu`를 추가하세요.
3. **Settings → Secrets and variables → Actions** 에 시크릿 4개를 등록합니다: `ZENML_API_KEY`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `RELOAD_TOKEN`(6장 추론 서버의 값).
4. **Settings → Environments** 에서 `production` 환경을 만듭니다.
5. 학습 관련 파일을 바꿔 push하면 `pipeline` 워크플로가 돌고, 요약에 나온 버전으로 **Actions → deploy → Run workflow** 를 실행하면 배포됩니다.
