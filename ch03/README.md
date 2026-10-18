# 3장. 데이터 관리 자동화

프로젝트 A(꽃 사진 5종 분류)의 학습 데이터를 DVC로 버전 관리하고, 품질 검사와 전처리를 파이프라인으로 자동화합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flowers/src/download.py` | 원본 데이터 내려받기 (체크섬 검증) | 03-1 |
| `flowers/src/validate.py` | 손상·크기·중복·라벨 충돌 검사, 품질 게이트 | 03-3 |
| `flowers/src/prepare.py` | 중복 묶음을 고려한 층화 분할, 크기 맞추기 | 03-4 |
| `flowers/dvc.yaml`, `flowers/params.yaml` | DVC 파이프라인과 파라미터 | 03-4 |

DVC 메타데이터(`.dvc/`, `data/raw.dvc`, `dvc.lock`)는 저장소에 넣지 않았습니다. 03-2를 따라 직접 만드세요.

## 빠른 실행

```bash
# 2장의 MLflow 서버(RustFS 포함)가 실행 중이어야 합니다 (dvc 버킷 사용)
cd flowers
uv sync
uv run python src/download.py
uv run dvc init --subdir
uv run dvc add data/raw
uv run dvc remote add -d rustfs s3://dvc/flowers
uv run dvc remote modify rustfs endpointurl http://localhost:9000
uv run dvc remote modify --local rustfs access_key_id mlops-admin
uv run dvc remote modify --local rustfs secret_access_key mlops-secret-key
uv run dvc push
uv run dvc repro
```

## 데이터 출처

TensorFlow flower_photos (3,670장). 모든 사진은 CC-BY 2.0이며 촬영자 목록은 `data/raw/LICENSE.txt`에 있습니다.
