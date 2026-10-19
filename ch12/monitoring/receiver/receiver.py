"""Alertmanager 웹훅을 받아 기록하는 수신기 (12-4).

실제 운영에서는 이 자리에 메신저(Slack·Teams 등)나 메일이 온다. 여기서는 받은 알림을
한 줄씩 출력하고(docker compose logs alert-receiver) /data/alerts.jsonl에 남긴다.
"""

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

LOG = "/data/alerts.jsonl"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        channel = parse_qs(urlparse(self.path).query).get("channel", ["-"])[0]
        received = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(LOG, "a", encoding="utf-8") as f:
            for a in body["alerts"]:
                record = {"received": received, "channel": channel, "status": a["status"],
                          "alertname": a["labels"].get("alertname"), "severity": a["labels"].get("severity"),
                          "service": a["labels"].get("service") or a["labels"].get("project"),
                          "summary": a["annotations"].get("summary"), "startsAt": a["startsAt"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{channel}] {a['status']:8} {record['alertname']} — {record['summary']}", flush=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):            # 요청마다 찍히는 접속 기록은 끈다
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
