"""모델 계보(lineage) 조회와 점검 (14-1). 프로젝트 A·B·C가 MLflow 서버 하나를 같이 쓴다.

    uv run python lineage.py show flowers-classifier@champion
    uv run python lineage.py show plate-recognizer/2
    uv run python lineage.py audit                 # 모든 등록 모델의 champion 점검. 빠진 답이 있으면 종료 코드 1
    uv run python lineage.py tag-experiments       # lineage.yaml의 projects대로 실험에 project 태그
    uv run python lineage.py experiments           # 프로젝트별 실험·실행 수
    uv run python lineage.py retire flowers-classifier-onnx "이유"   # champion 별칭을 떼고 은퇴 표시 (버전은 남긴다)

질문과 키는 lineage.yaml에 있다. 변환본(ONNX)은 원본 모델의 답을 물려받지만, 승인은 물려받지 않는다 —
운영에 올린 결정은 버전마다 따로 내린 것이기 때문이다.
"""

import argparse
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException

CONTRACT = yaml.safe_load(open(Path(__file__).with_name("lineage.yaml"), encoding="utf-8"))
client = mlflow.MlflowClient()


def resolve(ref: str):
    """'이름@별칭' 또는 '이름/버전' → 모델 버전."""
    if "@" in ref:
        return client.get_model_version_by_alias(*ref.split("@"))
    return client.get_model_version(*ref.split("/"))


def record(mv) -> dict:
    """학습 실행의 파라미터·태그 위에 모델 버전 태그를 겹친다 (같은 키면 버전 태그가 이긴다)."""
    rec = {}
    if mv.run_id:
        try:
            run = client.get_run(mv.run_id)
        except MlflowException:
            run = None
        if run:
            rec |= run.data.params | {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
            rec["_run"] = f"{client.get_experiment(run.info.experiment_id).name} / run {mv.run_id[:8]}"
    return rec | mv.tags


def answers(mv, depth: int = 0) -> dict:
    """질문 → (키, 값, 기대와 맞는가, 어디서 왔나)."""
    rec, out = record(mv), {}
    for q, spec in CONTRACT["questions"].items():
        key = next((k for k in spec["keys"] if k in rec), None)
        if key:
            out[q] = (key, rec[key], spec.get("expect") in (None, rec[key]), "")
    if "source_model" in rec and depth < 3:          # 변환본: 원본의 답을 물려받는다
        src = client.get_model_version(rec["source_model"], rec["source_version"])
        for q, a in answers(src, depth + 1).items():
            if q not in out and q != "approval":
                out[q] = a[:3] + (f"← {rec['source_model']} v{rec['source_version']}",)
    return out


def shown(key: str, value: str) -> str:
    if key == "approved_by":
        return "<승인자>"                             # 계정 이름은 화면에 내지 않는다 (13장)
    if key in ("git_commit", "base_revision") or key.endswith("_md5") or key.endswith("run_id"):
        return value[:12]
    return value


def show(ref: str) -> None:
    mv = resolve(ref)
    rec, ans = record(mv), answers(mv)
    print(f"{mv.name} v{mv.version}  ({rec.get('_run', '학습 실행 기록 없음')})")
    for q, spec in CONTRACT["questions"].items():
        if q in ans:
            key, value, ok, where = ans[q]
            print(f"  {spec['label']}: {key}={shown(key, value)}{'' if ok else '  ← 기대값 ' + spec['expect']} {where}")
        else:
            print(f"  {spec['label']}: (없음)")
    for key, label in CONTRACT["links"].items():
        if key in rec:
            value = {"source_version": f"{rec.get('source_model')} v{rec[key]}",
                     "crops_from_detector": f"plate-detector v{rec[key]}"}.get(key, shown(key, rec[key]))
            print(f"  → {label}: {value}")


def audit() -> int:
    questions = CONTRACT["questions"]
    print("모델 / champion | " + " | ".join(s["label"] for s in questions.values()))
    failed = 0
    for rm in client.search_registered_models():
        if "champion" not in rm.aliases:
            continue
        ans = answers(client.get_model_version(rm.name, rm.aliases["champion"]))
        cells = []
        for q in questions:
            if q not in ans:
                cells.append("없음")
            elif not ans[q][2]:
                cells.append("불일치")
            else:
                cells.append("OK" + (" (원본)" if ans[q][3] else ""))
        failed += any(c in ("없음", "불일치") for c in cells)
        print(f"{rm.name} v{rm.aliases['champion']} | " + " | ".join(cells))
        stale = stale_source(client.get_model_version(rm.name, rm.aliases["champion"]))
        if stale:
            failed += 1
            print(f"  └ {stale}")
    return 1 if failed else 0


def stale_source(mv) -> str:
    """변환본이 원본의 지금 champion에서 만들어졌는가. 원본만 새로 배포하고 변환을 잊으면 두 서빙이 갈라진다."""
    if "source_model" not in mv.tags:
        return ""
    src, ver = mv.tags["source_model"], mv.tags["source_version"]
    champion = client.get_registered_model(src).aliases.get("champion")
    return "" if champion == ver else f"원본 {src} v{ver}에서 변환 — 원본의 지금 champion은 v{champion}"


def tag_experiments() -> None:
    owner = {name: project for project, names in CONTRACT["projects"].items() for name in names}
    for exp in client.search_experiments():
        if exp.name in owner:
            client.set_experiment_tag(exp.experiment_id, "project", owner[exp.name])
        else:
            print(f"projects에 없는 실험: {exp.name}")


def experiments() -> None:
    for project in CONTRACT["projects"]:
        exps = client.search_experiments(filter_string=f"tags.project = '{project}'")
        runs = sum(len(client.search_runs([e.experiment_id], max_results=1000)) for e in exps)
        print(f"{project}: 실험 {len(exps)}개, 실행 {runs}개")


def retire(name: str, reason: str) -> None:
    """운영에서 뺀다. 버전은 지우지 않는다 — 지난 기록이 그 버전을 가리키고 있다."""
    if "champion" in client.get_registered_model(name).aliases:
        client.delete_registered_model_alias(name, "champion")
    client.set_registered_model_tag(name, "status", "retired")
    client.set_registered_model_tag(name, "retired_reason", reason)
    print(f"{name}: champion 별칭 삭제, status=retired")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show").add_argument("ref")
    for cmd in ("audit", "tag-experiments", "experiments"):
        sub.add_parser(cmd)
    r = sub.add_parser("retire")
    r.add_argument("name")
    r.add_argument("reason")
    args = p.parse_args()
    if args.cmd == "show":
        show(args.ref)
    elif args.cmd == "retire":
        retire(args.name, args.reason)
    elif args.cmd == "tag-experiments":
        tag_experiments()
    elif args.cmd == "experiments":
        experiments()
    else:
        sys.exit(audit())


if __name__ == "__main__":
    main()
