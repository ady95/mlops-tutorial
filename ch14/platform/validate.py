"""최종 시스템 점검 (14-5). 하나라도 실패하면 종료 코드 1.

    uv run python validate.py
    uv run python validate.py --config validate.local.yaml     # 서버가 여러 대일 때 주소만 바꾼 사본

1. 서비스가 살아 있는가                    4. 드리프트 검사가 멈추지 않았는가
2. 서빙 중인 버전 = 레지스트리 champion인가   5. 묵음 처리되지 않은 알림이 울리고 있지 않은가
3. 되돌릴 버전(previous)이 있는가          6. 운영 모델의 계보가 빠짐없는가 (14-1 lineage.py audit)
"""

import argparse
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException

client = mlflow.MlflowClient()
CONFIG = {}
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"[{'OK' if ok else '실패'}] {name}{' — ' + detail if detail else ''}")


def get(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body[:1] in (b"{", b"[") else body.decode()


def alias(name: str, which: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, which).version
    except MlflowException:
        return None


def prom(query: str) -> list:
    return get(f"{CONFIG['prometheus']}/api/v1/query?" + urllib.parse.urlencode({"query": query}))["data"]["result"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).with_name("validate.yaml")))
    CONFIG.update(yaml.safe_load(open(p.parse_args().config, encoding="utf-8")))
    for name, url in CONFIG["health"].items():
        try:
            get(url)
            check(f"{name} 응답", True)
        except OSError as e:
            check(f"{name} 응답", False, str(e))

    for server, spec in CONFIG["serving"].items():
        try:
            health = get(spec["url"])
        except OSError as e:
            check(f"{server} 응답", False, str(e))
            continue
        for key, model in spec.get("models", {}).items():
            champion, previous = alias(model, "champion"), alias(model, "previous")
            check(f"{server}: {model} 서빙 v{health.get(key)} = champion v{champion}", health.get(key) == champion)
            check(f"{server}: {model} 되돌릴 버전(previous)", previous is not None, f"v{previous}")
        for key, prompt in spec.get("prompts", {}).items():
            production = mlflow.genai.load_prompt(f"prompts:/{prompt}@production").version
            check(f"{server}: {prompt} 서빙 v{health.get(key)} = production v{production}",
                  str(health.get(key)) == str(production))

    for job, hours in CONFIG["drift_max_age_hours"].items():
        found = prom(f'time() - push_time_seconds{{job="{job}"}}')
        age = float(found[0]["value"][1]) / 3600 if found else None
        check(f"드리프트 검사 {job}", age is not None and age <= hours,
              f"마지막 결과 {age:.1f}시간 전 (기준 {hours}시간)" if age is not None else "기록 없음")

    down = [r["metric"].get("job") for r in prom("up == 0")]
    check("Prometheus 수집 대상 모두 up", not down, ", ".join(down))
    firing = get(f"{CONFIG['alertmanager']}/api/v2/alerts?active=true&silenced=false&inhibited=false")
    check("울리는 알림 없음 (묵음 제외)", not firing, ", ".join(a["labels"]["alertname"] for a in firing))
    silences = [s for s in get(f"{CONFIG['alertmanager']}/api/v2/silences") if s["status"]["state"] == "active"]
    for s in silences:
        names = ",".join(m["value"] for m in s["matchers"] if m["name"] == "alertname")
        print(f"  (묵음 중: {names}, {s['endsAt'][:16]}까지)")

    audit = subprocess.run([sys.executable, str(Path(__file__).with_name("lineage.py")), "audit"],
                           capture_output=True, text=True)
    check("운영 모델 계보 (lineage.py audit)", audit.returncode == 0)

    print(f"\n{sum(results)}/{len(results)} 통과")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
