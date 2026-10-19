"""LLM 배포 게이트: 평가 결과로 판정하고, 통과하면 어댑터를 레지스트리에 등록한다 (09-5).

판정에 쓰는 실행 (ch09-evaluation 실험):
    <후보>-holdout  eval_llm --data holdout 결과 + judge 결과 (보류 세트: 개발 중에 한 번도 보지 않은 시험지)
    <후보>-indist   eval_llm --data indist 결과 (8장 테스트 세트, 회귀 확인)

실행:
    uv run python -m src.gate r16v2-v2 --train-run <학습 실행 ID> --prompt prompts:/flower-support-extract/2
    uv run python -m src.gate r16v2-v2 --train-run <학습 실행 ID> --prompt prompts:/flower-support-extract/2 --approve
"""

import argparse
import getpass
import json
from datetime import datetime

import mlflow
import yaml

MODEL_NAME = "flower-support-adapter"
SLICES = ["long", "multi", "boundary", "short", "order_trap", "injection"]


def judge(m: dict, indist: dict, champion: dict | None, gate: dict) -> list[tuple[str, bool]]:
    """배포 기준 판정 (MLflow 없이 도는 순수 함수)."""
    checks = [
        (f"JSON 형식 유효 {m['eval_json_valid']:.3f} >= {gate['min_json_valid']}", m["eval_json_valid"] >= gate["min_json_valid"]),
        (f"보류 세트 다섯 필드 정답 {m['eval_all_fields']:.3f} >= {gate['min_all_fields']}",
         m["eval_all_fields"] >= gate["min_all_fields"]),
    ]
    for s in SLICES:
        v = m[f"slice_{s}"]
        checks.append((f"  조각 {s} {v:.2f} >= {gate['min_slice_all_fields']}", v >= gate["min_slice_all_fields"]))
    checks.append((f"요약 심판 통과 {m['judge_summary_pass']:.3f} >= {gate['min_summary_pass']}",
                   m["judge_summary_pass"] >= gate["min_summary_pass"]))
    checks.append((f"학습 분포(8장 테스트) 정답 {indist['indist_all_fields']:.3f} >= {gate['min_indist_all_fields']}",
                   indist["indist_all_fields"] >= gate["min_indist_all_fields"]))
    if champion is not None:
        c = float(champion["eval_all_fields"])
        checks.append((f"현재 champion({c:.3f}) 대비 하락 <= {gate['max_drop_vs_champion']}",
                       c - m["eval_all_fields"] <= gate["max_drop_vs_champion"]))
    return checks


def run_metrics(name: str) -> tuple[str, dict]:
    runs = mlflow.search_runs(experiment_names=["ch09-evaluation"], output_format="list", max_results=1,
                              filter_string=f"attributes.run_name = '{name}' AND attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"])
    if not runs:
        raise SystemExit(f"평가 실행 '{name}'이 없습니다. eval_llm.py와 judge.py를 먼저 실행하세요.")
    return runs[0].info.run_id, runs[0].data.metrics


def champion_tags() -> tuple[str | None, dict | None]:
    try:
        mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        return mv.version, mv.tags
    except mlflow.exceptions.MlflowException:
        return None, None


def register(train_run: str, prompt_uri: str, metrics: dict, eval_run: str) -> str:
    """어댑터를 MLflow 모델(transformers 형식)로 기록하고 레지스트리에 올린다. 베이스는 이름과 리비전만 기록된다."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(open("params.yaml", encoding="utf-8"))["model"]
    tok = AutoTokenizer.from_pretrained(cfg["name"], revision=cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["name"], revision=cfg["revision"], dtype=torch.bfloat16,
                                                device_map="cpu")
    adapter = mlflow.artifacts.download_artifacts(run_id=train_run, artifact_path="adapter")
    model = PeftModel.from_pretrained(base, adapter)
    with mlflow.start_run(run_id=train_run):     # 학습 실행에 모델을 붙인다 (어댑터 130MB만 저장된다)
        info = mlflow.transformers.log_model(transformers_model={"model": model, "tokenizer": tok}, name="model",
                                             task="text-generation", prompts=[prompt_uri])
    tags = {k: f"{v:.4f}" for k, v in metrics.items() if k.startswith(("eval_", "judge_", "slice_"))}
    tags |= {"prompt_uri": prompt_uri, "train_run": train_run, "eval_run": eval_run,
             "base_revision": cfg["revision"]}
    mv = mlflow.register_model(info.model_uri, MODEL_NAME, tags=tags)
    return mv.version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", help="평가 실행 이름 (예: r16v2-v2)")
    parser.add_argument("--train-run", required=True, help="어댑터를 학습한 MLflow 실행 ID")
    parser.add_argument("--prompt", required=True, help="함께 서빙할 프롬프트 (prompts:/이름/버전)")
    parser.add_argument("--approve", action="store_true", help="통과한 후보를 champion으로 승격 (사람의 승인)")
    args = parser.parse_args()

    gate = yaml.safe_load(open("params.yaml", encoding="utf-8"))["gate"]
    eval_run, m = run_metrics(f"{args.candidate}-holdout")
    _, indist = run_metrics(f"{args.candidate}-indist")
    champ_version, champ = champion_tags()
    checks = judge(m, indist, champ, gate)
    passed = all(ok for _, ok in checks)
    print(f"후보 {args.candidate} (프롬프트 {args.prompt}, 현재 champion v{champ_version})")
    for text, ok in checks:
        print(f"  [{'OK' if ok else '불가'}] {text}")
    print(f"판정: {'통과' if passed else '불가 — 등록하지 않습니다'}")
    json.dump({"candidate": args.candidate, "passed": passed, "checks": [{"rule": t, "ok": ok} for t, ok in checks]},
              open("gate_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if not passed:
        raise SystemExit(1)

    client = mlflow.MlflowClient()
    version = register(args.train_run, args.prompt, m, eval_run)
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"등록: {MODEL_NAME} v{version} → challenger")
    if args.approve:   # 판정은 코드가, 승격은 사람이 (05-4) — 모델과 프롬프트를 함께 옮긴다
        if champ_version:
            client.set_registered_model_alias(MODEL_NAME, "previous", champ_version)
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
        prompt_name, prompt_version = args.prompt.removeprefix("prompts:/").split("/")
        mlflow.genai.set_prompt_alias(prompt_name, alias="production", version=int(prompt_version))
        client.set_model_version_tag(MODEL_NAME, version, "approved_by", getpass.getuser())
        client.set_model_version_tag(MODEL_NAME, version, "approved_at", datetime.now().isoformat(timespec="seconds"))
        print(f"승격: {MODEL_NAME} v{version} → champion, 프롬프트 {prompt_name} v{prompt_version} → production")
