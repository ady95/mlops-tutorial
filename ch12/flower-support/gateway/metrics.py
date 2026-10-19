"""게이트웨이의 Prometheus 지표 (12-1).

11장의 트레이스는 요청 하나하나를 자세히 남기고, 이 지표는 그것을 시간에 따라 센다.
트레이스는 "이 요청은 왜 이랬나", 지표는 "지금 평소와 다른가"를 답한다.
"""

from prometheus_client import Counter, Gauge, Histogram

LABELS = ["model_version", "prompt_version"]
REQUESTS = Counter("llm_requests_total", "처리한 요청 수", LABELS + ["channel", "status"])
LATENCY = Histogram("llm_request_seconds", "요청 처리 시간(초) — 작업 스레드를 받은 뒤부터", LABELS,
                    buckets=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0])
# 아래 둘은 미들웨어에서 잰다 (12-2). 동기 함수는 작업 스레드(기본 40개)가 비어야 시작하므로,
# 몰릴 때는 스레드를 기다리는 시간이 LATENCY에 잡히지 않는다.
IN_FLIGHT = Gauge("llm_in_flight_requests", "게이트웨이가 받아 놓고 아직 답하지 않은 요청 수 (스레드 대기 포함)")
TOTAL_LATENCY = Histogram("llm_request_total_seconds", "요청을 받은 때부터 답할 때까지(초), 스레드 대기 포함",
                          buckets=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0])
TOKENS = Counter("llm_tokens_total", "사용한 토큰 수", LABELS + ["kind"])          # kind: input / output
SIGNALS = Counter("llm_signal_total", "운영 신호가 켜진 요청 수 (11-4)", LABELS + ["signal"])
CONFIDENCE = Histogram("llm_category_confidence", "분류 확신도 (category 첫 토큰 확률)", LABELS,
                       buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0])
