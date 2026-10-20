"""Prometheus 지표 (12-1). /metrics로 내보내고, Prometheus가 15초마다 가져간다.

서비스 지표(요청 수, 지연)에 더해 모델 지표(예측 클래스, 확신도)와 입력 지표(사진 밝기)를 남긴다.
입력과 출력의 분포가 평소와 달라지는 것이 드리프트의 첫 신호다 (12-3).
주의: uvicorn 작업자를 여러 개(--workers) 쓰면 프로세스마다 지표가 따로 쌓인다. 이 책은 작업자 1개 기준이다.
"""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("flowers_requests_total", "요청 수", ["endpoint", "status"])
LATENCY = Histogram("flowers_predict_seconds", "예측 처리 시간(초)",
                    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0])
PREDICTIONS = Counter("flowers_predictions_total", "예측 결과", ["predicted_class", "model_version"])
CONFIDENCE = Histogram("flowers_prediction_confidence", "가장 높은 클래스의 확률",
                       buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
BRIGHTNESS = Histogram("flowers_input_brightness", "입력 사진의 평균 밝기 (0~1)",
                       buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MODEL_VERSION = Gauge("flowers_model_version", "지금 서비스 중인 모델 버전")
