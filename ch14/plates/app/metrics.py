"""plate API의 Prometheus 지표 (14-2). 모델이 둘이면 지연도, 버전도 모델(단계)별로 나눠 남긴다."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("plates_requests_total", "요청 수", ["status"])
STAGE_SECONDS = Histogram("plates_stage_seconds", "단계별 처리 시간(초)", ["stage"],
                          buckets=[0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5])
FORMAT_OK = Counter("plates_format_total", "인식 결과가 번호판 형식에 맞는가",
                    ["ok", "detector_version", "recognizer_version"])
MODEL_VERSION = Gauge("plates_model_version", "지금 서비스 중인 모델 버전", ["model"])
