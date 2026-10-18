"""ZenML 파이프라인 단계 (07-2)."""

import json
import subprocess

import mlflow
import yaml
from zenml import get_step_context, log_metadata, step

from src import registry, training


@step(enable_cache=False)   # 데이터가 바뀌었는지는 DVC가 판단한다 (dvc repro가 알아서 건너뜀)
def prepare_data() -> dict:
    subprocess.run(["dvc", "pull"], check=True)    # 깨끗한 폴더(CI 러너)에서도 RustFS에서 데이터를 받는다
    subprocess.run(["dvc", "repro"], check=True)
    version = training.data_version()
    log_metadata(metadata=version)
    return version


@step
def train_model(data: dict, config: dict) -> str:
    run_id = get_step_context().pipeline_run.id
    result = training.train(config, experiment="ch07-pipeline",
                            extra_tags={"zenml_pipeline_run_id": str(run_id)})
    log_metadata(metadata={k: v for k, v in result.items() if k != "run_id"} | {"mlflow_run_id": result["run_id"]})
    return result["run_id"]


@step
def register_model(mlflow_run_id: str) -> str:
    version = registry.register(mlflow_run_id)
    log_metadata(metadata={"model_version": version})
    return version


@step
def evaluate_model(version: str) -> dict:
    metrics = registry.evaluate(version)
    log_metadata(metadata=metrics)
    return metrics


@step
def gate(version: str, metrics: dict) -> dict:
    """승인 기준 판정. 통과하면 challenger 별칭까지만 — champion 승격은 사람이 한다 (07-5)."""
    rules = yaml.safe_load(open("params.yaml"))["gate"]
    client = mlflow.MlflowClient()
    tags = client.get_model_version(registry.NAME, version).tags
    champ_version, champ = registry.champion_tags()
    if champ_version == version:
        champ = None
    checks = registry.judge(tags, champ, rules)
    passed = all(ok for _, ok in checks)
    if passed:
        client.set_registered_model_alias(registry.NAME, "challenger", version)
    result = {"version": version, "passed": passed, "champion_version": champ_version,
              "checks": [{"rule": text, "ok": ok} for text, ok in checks]}
    log_metadata(metadata={"passed": passed})
    json.dump(result, open("gate_result.json", "w"), ensure_ascii=False, indent=2)   # CI가 읽는 결과
    for c in result["checks"]:
        print(f"  [{'OK' if c['ok'] else '불가'}] {c['rule']}")
    print(f"판정: {'통과 — challenger' if passed else '불가'} (v{version})")
    return result
