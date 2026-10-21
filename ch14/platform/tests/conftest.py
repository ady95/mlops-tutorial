import os

# 모듈을 불러올 때 MlflowClient를 만든다. 테스트는 서버에 접속하지 않으므로 주소만 채워 둔다.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:1")
