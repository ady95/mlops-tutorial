"""꽃 분류 학습 파이프라인 (07-2, 07-3).

실행 (프로젝트 루트에서):
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export ZENML_STORE_URL=http://localhost:8080 ZENML_STORE_API_KEY=...
    uv run python -m pipeline.run
"""

import yaml
from zenml import pipeline

from pipeline.steps import evaluate_model, gate, prepare_data, register_model, train_model


# 캐시를 끈다: 학습·등록·평가·판정은 MLflow와 레지스트리를 바꾸는 부수효과가 있고, ZenML이 보지 못하는
# 상태(Git 커밋 상태, 현재 champion)에 따라 결과가 달라진다. "바뀐 것만 다시"는 데이터 단계에서 DVC가 맡는다.
@pipeline(name="flowers_training", enable_cache=False)
def flowers_training(config: dict):
    data = prepare_data()
    run_id = train_model(data, config)
    version = register_model(run_id)
    metrics = evaluate_model(version)
    gate(version, metrics)


if __name__ == "__main__":
    config = yaml.safe_load(open("params.yaml"))["train"]
    flowers_training(config=config)
