"""서빙 중인 모델(OpenAI 호환 API)로 보류 세트를 다시 채점하고, 게이트에서 본 오프라인 결과와 비교한다 (10-2, 10-3).

"게이트를 통과한 것"과 "실제로 서빙되는 것"이 같은지 확인하는 단계다.

실행:
    uv run python -m src.eval_served --model flower-support-v1 --offline eval/outputs/r16v2-v2-holdout.jsonl
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import mlflow
from openai import AsyncOpenAI

from src.schema import parse_output, score

KEYS = ["category", "urgency", "sentiment", "order_id", "product"]
SLICE_KEYS = {"길고 산만한 문의": "long", "여러 요청": "multi", "경계 사례": "boundary",
              "짧은 문의": "short", "주문번호 함정": "order_trap", "지시 주입": "injection",
              "학습 분포": "indist", "새 상품": "new_product"}

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8001/v1")
parser.add_argument("--model", required=True, help="vLLM에 올린 모델(또는 어댑터) 이름")
parser.add_argument("--prompt", default="prompts:/flower-support-extract@production")
parser.add_argument("--data", default="eval/holdout.jsonl")
parser.add_argument("--offline", help="같은 데이터의 오프라인 평가 출력 (eval_llm.py) — 있으면 문항별로 비교")
parser.add_argument("--concurrency", type=int, default=16)
parser.add_argument("--name", help="결과 이름 (eval/outputs/<이름>.jsonl, MLflow 실행 이름). judge.py로 요약을 채점할 수 있다")
args = parser.parse_args()

prompt = mlflow.genai.load_prompt(args.prompt)
client = AsyncOpenAI(base_url=args.base_url, api_key="unused")
rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
if "prompt" in rows[0]:   # 8장 테스트 세트 형식(prompt/completion)이면 문의와 정답만 꺼낸다 — 회귀 확인용
    rows = [{"id": r["id"], "slice": "학습 분포", "inquiry": r["prompt"][1]["content"],
             "gold": json.loads(r["completion"][0]["content"])} for r in rows]
PREFIX = "indist" if rows[0]["slice"] == "학습 분포" else "eval"


async def one(sem, r):
    async with sem:
        resp = await client.chat.completions.create(
            model=args.model, messages=prompt.format(inquiry=r["inquiry"]), temperature=0, max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})    # 학습·평가와 같은 형식 (08-3)
        text = resp.choices[0].message.content
        pred = parse_output(text)
        s = score(pred, r["gold"] | {"summary": None})
        return {"id": r["id"], "slice": r["slice"], "inquiry": r["inquiry"], "output": text, "pred": pred,
                "summary": (pred or {}).get("summary"), **s,
                "all_fields": all(s[k] for k in KEYS)}


async def main():
    sem = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    results = await asyncio.gather(*(one(sem, r) for r in rows))
    elapsed = time.perf_counter() - start
    n = len(results)
    print(f"{args.model} (프롬프트 v{prompt.version}): {n}건 {elapsed:.1f}초")
    for k in ["json_valid", "schema_valid", "all_fields"]:
        print(f"  {k:14s} {sum(r[k] for r in results) / n:.4f}")
    if args.offline:
        off = {o["id"]: o for o in map(json.loads, open(args.offline, encoding="utf-8"))}
        same_fields = sum({k: r["pred"].get(k) for k in KEYS} == {k: (parse_output(off[r["id"]]["output"]) or {}).get(k)
                                                                   for k in KEYS} for r in results if r["pred"])
        same_text = sum(r["output"].strip() == off[r["id"]]["output"].strip() for r in results)
        print(f"  오프라인과 다섯 필드가 같은 문항 {same_fields}/{n}, 출력 글자까지 같은 문항 {same_text}/{n}")
    if args.name:   # 9장 게이트·심판과 같은 형식으로 남긴다
        by_slice = defaultdict(list)
        for r in results:
            by_slice[r["slice"]].append(r["all_fields"])
        metrics = {f"{PREFIX}_{k}": sum(r[k] for r in results) / n for k in ["json_valid", "schema_valid", *KEYS, "all_fields"]}
        metrics |= {f"slice_{SLICE_KEYS[s]}": sum(v) / len(v) for s, v in by_slice.items()}
        out_path = Path("eval/outputs") / f"{args.name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for r in results:
                f.write(json.dumps({k: v for k, v in r.items() if k != "pred"}, ensure_ascii=False) + "\n")
        mlflow.set_experiment("ch09-evaluation")
        with mlflow.start_run(run_name=args.name) as run:
            mlflow.set_tags({"served_model": args.model, "prompt_uri": args.prompt, "data": Path(args.data).stem,
                             "serving": "vllm"})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out_path))
        out_path.with_suffix(".run_id").write_text(run.info.run_id)
        print("  " + ", ".join(f"{k.removeprefix('slice_')} {v:.2f}" for k, v in metrics.items() if k.startswith("slice_")))


asyncio.run(main())
