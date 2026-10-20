"""운영 도구 — 모두 읽기 전용 (13-2). Prometheus 지표, 알림 이력, 수집 대상 상태, 서비스 로그.

설정: PROMETHEUS_URL (기본 http://localhost:9090), services.yaml (서비스 이름 → 로그를 읽는 명령)
"""

import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from agent.core import tool

PROM = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
DROP_LABELS = {"instance", "hostname", "Hostname", "UUID", "pci_bus_id", "DCGM_FI_DRIVER_VERSION",
               "device", "modelName", "container", "pod", "namespace"}     # 판단에 쓸모없고 내부 정보만 담은 라벨


def ts(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def labels(metric: dict) -> dict:
    return {k: v for k, v in metric.items() if k not in DROP_LABELS}


def prom(path: str, **params) -> list:
    r = httpx.get(f"{PROM}/api/v1/{path}", params=params, timeout=30)
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(body.get("error", r.text[:200]))
    return body["data"]["result"] if "result" in body.get("data", {}) else body["data"]


@tool("PromQL 식을 한 시각에 계산한다. at을 비우면 지금.",
      promql={"type": "string"}, at={"type": "string", "description": "ISO 시각, 예: 2026-10-19T11:00:00Z"})
def query(promql: str, at: str = "") -> list:
    params = {"query": promql} | ({"time": ts(at)} if at else {})
    return [{"labels": labels(r["metric"]), "value": round(float(r["value"][1]), 4)} for r in prom("query", **params)]


@tool("PromQL 식의 시간 흐름. 시계열마다 최소·최대와 고르게 고른 점 12개 이하를 돌려준다.",
      promql={"type": "string"}, start={"type": "string", "description": "ISO 시각"},
      end={"type": "string", "description": "ISO 시각"}, step_seconds={"type": "integer"})
def query_range(promql: str, start: str, end: str, step_seconds: int = 60) -> list:
    out = []
    for r in prom("query_range", query=promql, start=ts(start), end=ts(end), step=step_seconds)[:20]:
        pts = [(float(t), float(v)) for t, v in r["values"]]
        every = max(1, len(pts) // 12)
        out.append({"labels": labels(r["metric"]), "min": round(min(v for _, v in pts), 4),
                    "max": round(max(v for _, v in pts), 4),
                    "points": [[iso(t)[11:19], round(v, 4)] for t, v in pts[::every]]})
    return out


@tool("구간 안에서 알림(ALERTS)이 대기·발생한 기록. 알림 이름, 라벨, 상태, 처음·마지막 시각.",
      start={"type": "string"}, end={"type": "string"})
def alert_history(start: str, end: str) -> list:
    rows = prom("query_range", query="ALERTS", start=ts(start), end=ts(end), step=30)
    out = []
    for r in rows:
        m = labels(r["metric"])
        times = [float(t) for t, _ in r["values"]]
        out.append({"alert": m.pop("alertname"), "state": m.pop("alertstate"), "labels": m,
                    "first": iso(min(times)), "last": iso(max(times))})
    return sorted(out, key=lambda x: x["first"])


@tool("Prometheus가 지금 수집하는 대상과 상태(up/down), 마지막 수집 오류.")
def targets() -> list:
    data = prom("targets")
    return [{"job": t["labels"]["job"], "server": t["labels"].get("server", ""), "health": t["health"],
             "last_error": t.get("lastError", "")} for t in data["activeTargets"]]


SERVICES = Path(os.environ.get("SERVICES_FILE", "services.yaml"))


@tool("서비스 컨테이너 로그의 마지막 줄들. pattern을 주면 그 글자가 든 줄만.",
      service={"type": "string", "description": "services.yaml에 적은 이름 (예: llm-gateway, flowers-api)"},
      since={"type": "string", "description": "ISO 시각부터"}, until={"type": "string", "description": "ISO 시각까지"},
      pattern={"type": "string"}, max_lines={"type": "integer"})
def service_logs(service: str, since: str = "", until: str = "", pattern: str = "", max_lines: int = 60) -> str:
    commands = yaml.safe_load(SERVICES.read_text(encoding="utf-8"))
    if service not in commands:
        return f"모르는 서비스: {service}. 가능한 이름: {', '.join(commands)}"
    cmd = shlex.split(commands[service]) + (["--since", since] if since else []) + (["--until", until] if until else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    lines = (proc.stdout + proc.stderr).splitlines()
    if pattern:
        lines = [line for line in lines if pattern.lower() in line.lower()]
    return "\n".join(lines[-max_lines:]) or "(해당하는 로그 없음)"


OPS_TOOLS = [query, query_range, alert_history, targets, service_logs]
