"""모델별 승인 판정 (14-2). 판정 기준은 언제나 '끝단 정확도' — 짝 모델의 현재 champion과 이어 붙여서 잰다.

    uv run python -m src.registry gate recognizer 2        # 후보 + 짝 champion vs 지금 champion + 짝 champion
    uv run python -m src.registry gate detector 1 --partner 1   # 맨 처음 한 번: 짝 champion이 아직 없을 때

통과하면 challenger 별칭만 붙인다. champion으로 올리고 서버에 반영하는 일은 src/deploy.py가 한다 (7장과 같은 나눔).

검출 모델 후보를 혼자 재면 IoU가 올라도, 그 상자로 자른 조각을 지금 인식 모델이 못 읽으면 서비스는 나빠진다.
"""

import argparse
import json

import mlflow
from mlflow.exceptions import MlflowException

from src.evaluate import NAMES, end_to_end
from src.models import load_split

OTHER = {"detector": "recognizer", "recognizer": "detector"}
MIN_GAIN = 0.0            # 끝단 정확도가 champion보다 낮으면 떨어뜨린다
MAX_SLOWDOWN = 1.5        # 그 모델 단계의 지연이 1.5배를 넘으면 떨어뜨린다
STAGE_MS = {"detector": "det_ms_per_image", "recognizer": "rec_ms_per_image"}


def alias_version(client, kind: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(NAMES[kind], alias).version
    except MlflowException:
        return None


def verified_pairs(client) -> set[tuple[str, str]]:
    """판정을 통과한 (검출 버전, 인식 버전) 조합. 버전마다 gate가 남긴 태그에서 모은다."""
    pairs = set()
    for kind in NAMES:
        for mv in client.search_model_versions(f"name='{NAMES[kind]}'"):
            if mv.tags.get("gate_passed") == "True":
                pair = {kind: mv.version, OTHER[kind]: mv.tags["gate_partner"].split(" v")[-1]}
                pairs.add((pair["detector"], pair["recognizer"]))
    return pairs


def gate(kind: str, version: str, data: str, partner: str | None = None) -> dict:
    client = mlflow.MlflowClient()
    rows = load_split(data, "test")
    partner = partner or alias_version(client, OTHER[kind], "champion")
    if partner is None:                                     # 맨 처음 한 번은 짝 버전을 --partner로 정해 준다
        raise SystemExit(f"{NAMES[OTHER[kind]]}에 champion이 없습니다. --partner로 짝 버전을 지정하세요")
    champion = alias_version(client, kind, "champion")
    pair = {kind: version, OTHER[kind]: partner}
    cand = end_to_end(pair["detector"], pair["recognizer"], rows)
    result = {"kind": kind, "candidate": version, "partner": f"{OTHER[kind]} v{partner}", "candidate_metrics": cand}
    reasons = []
    if champion:
        base_pair = {kind: champion, OTHER[kind]: partner}
        base = end_to_end(base_pair["detector"], base_pair["recognizer"], rows)
        result |= {"champion": champion, "champion_metrics": base}
        if cand["e2e_exact"] < base["e2e_exact"] + MIN_GAIN:
            reasons.append(f"끝단 정확도 {cand['e2e_exact']:.4f} < champion {base['e2e_exact']:.4f}")
        if cand[STAGE_MS[kind]] > base[STAGE_MS[kind]] * MAX_SLOWDOWN:
            reasons.append(f"{kind} 지연 {cand[STAGE_MS[kind]]:.1f}ms > champion의 {MAX_SLOWDOWN}배")
    result["passed"] = not reasons
    result["reasons"] = reasons
    client.set_model_version_tag(NAMES[kind], version, "gate_e2e_exact", f"{cand['e2e_exact']:.4f}")
    client.set_model_version_tag(NAMES[kind], version, "gate_partner", f"{OTHER[kind]} v{partner}")
    client.set_model_version_tag(NAMES[kind], version, "gate_passed", str(not reasons))
    if not reasons:
        client.set_registered_model_alias(NAMES[kind], "challenger", version)
    return result


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("kind", choices=list(NAMES))
    g.add_argument("version")
    g.add_argument("--data", default="data/scenes")
    g.add_argument("--partner", help="짝 모델 버전 (기본: 짝의 champion)")
    args = p.parse_args()
    print(json.dumps(gate(args.kind, args.version, args.data, args.partner), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
