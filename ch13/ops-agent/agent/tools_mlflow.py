"""실험 기록 도구 — 읽기 전용 (13-3). MLflow의 실험·실행·모델 레지스트리.

mlflow.* 시스템 태그(실행한 계정, 소스 경로 등)는 판단에 쓸모가 없고 내부 정보를 담으므로 넘기지 않는다.
"""

import mlflow

from agent.core import tool

client = mlflow.MlflowClient()


def run_row(r) -> dict:
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    for k in ("git_commit", "data_prepared_md5", "data_raw_md5"):
        if k in tags:
            tags[k] = tags[k][:8]
    return {"run_id": r.info.run_id[:8], "name": r.info.run_name, "status": r.info.status,
            "minutes": round((r.info.end_time - r.info.start_time) / 60000, 1) if r.info.end_time else None,
            "params": r.data.params, "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()}, "tags": tags}


@tool("실험 목록과 실행 수.")
def list_experiments() -> list:
    return [{"name": e.name, "runs": len(client.search_runs([e.experiment_id], max_results=1000))}
            for e in client.search_experiments() if e.name != "Default"]


@tool("한 실험의 실행들. filter는 MLflow 검색식(예: params.model = 'resnet18'), order_by 예: metrics.val_f1_macro DESC.",
      experiment={"type": "string"}, filter={"type": "string"}, order_by={"type": "string"},
      max_results={"type": "integer"})
def search_runs(experiment: str, filter: str = "", order_by: str = "", max_results: int = 30) -> list:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return f"없는 실험: {experiment}"
    runs = client.search_runs([exp.experiment_id], filter_string=filter, max_results=max_results,
                              order_by=[order_by] if order_by else None)
    return [run_row(r) for r in runs]


@tool("한 실행의 지표 하나가 에폭(단계)마다 어떻게 변했는지.",
      run_id={"type": "string", "description": "search_runs가 준 8자리 또는 전체 ID"}, key={"type": "string"})
def metric_history(run_id: str, key: str) -> list:
    if len(run_id) < 32:          # 8자리 접두로 전체 ID 찾기
        hits = [r.info.run_id for e in client.search_experiments()
                for r in client.search_runs([e.experiment_id], max_results=1000) if r.info.run_id.startswith(run_id)]
        if len(hits) != 1:
            return f"실행 ID를 하나로 찾지 못함: {run_id} ({len(hits)}개)"
        run_id = hits[0]
    return [[m.step, round(m.value, 4)] for m in client.get_metric_history(run_id, key)]


PERSON_TAGS = ("approved_by", "restored_by")      # 값은 사람(계정) 이름 — 있는지만 알면 된다 (13-5 실측: 서버 계정명이 들어 있었다)


@tool("등록 모델의 버전·별칭·태그(평가 점수, 학습 실행, 데이터 해시 등). 최신 버전이 먼저 나온다.", name={"type": "string"})
def model_versions(name: str) -> list:
    aliases = client.get_registered_model(name).aliases
    by_version = {}
    for alias, v in aliases.items():
        by_version.setdefault(v, []).append(alias)
    return [{"version": mv.version, "aliases": by_version.get(mv.version, []), "run_id": (mv.run_id or "")[:8],
             "description": mv.description,
             "tags": {k: ("<승인자>" if k in PERSON_TAGS else v) for k, v in mv.tags.items()}}
            for mv in client.search_model_versions(f"name = '{name}'")]


MLFLOW_TOOLS = [list_experiments, search_runs, metric_history, model_versions]
