# 10장. LLM 모델 배포와 추론 최적화

9장의 champion 어댑터를 vLLM으로 서빙하고, 양자화로 속도·메모리를 비교하고, 게이트웨이를 두어 요청을 받는 중에 업데이트·롤백합니다.

| 파일 | 내용 | 관련 절 |
|---|---|---|
| `flower-support/serving/compose.yaml` | vLLM(베이스 + 실행 중 어댑터 올리기) + 게이트웨이 | 10-1, 10-5 |
| `flower-support/src/adapter_for_vllm.py` | 어댑터 층 이름을 vLLM용으로 변경 (안 하면 오류 없이 무시됨) | 10-1 |
| `flower-support/src/client_demo.py` | OpenAI SDK로 vLLM 호출, 스트리밍 | 10-2 |
| `flower-support/src/eval_served.py` | 서빙 중인 모델로 보류 세트 재채점, 게이트 결과와 비교 | 10-2, 10-3 |
| `flower-support/quantize/` | GPTQ W4A16 변환 (llm-compressor 전용 환경) | 10-3 |
| `flower-support/src/bench_llm.py` | 부하 테스트: 처리량, TTFT, 지연, vLLM 지표 | 10-4 |
| `flower-support/gateway/` | 게이트웨이: champion + production 프롬프트로 응답, `/reload`, 적용 확인 | 10-5 |
| `flower-support/src/promote.py` | 승인·롤백 (모델 별칭과 프롬프트 별칭을 함께 이동) | 10-5 |
| `flower-support/src/traffic.py` | 업데이트·롤백 중 연속 요청과 버전 기록 | 10-5 |
| `flower-support/src/` 그 밖 | 8·9장 코드 (학습, 평가, 게이트) | 8~9장 |

데이터는 [`../ch08/dataset/`](../ch08/dataset/), [`../ch09/dataset/`](../ch09/dataset/)의 파일을 씁니다 (9장 README의 빠른 실행 참고).

## 빠른 실행

```bash
cd flower-support
# 9장까지 마쳐 레지스트리에 flower-support-adapter@champion, flower-support-extract@production이 있어야 합니다

# 1. vLLM + 게이트웨이
echo "RELOAD_TOKEN=$(openssl rand -hex 24)" > serving/.env
docker compose -f serving/compose.yaml up -d --build      # 첫 시작 5분 남짓
curl -s localhost:8002/health
curl -s -X POST localhost:8002/extract -H "Content-Type: application/json" -d '{"inquiry": "로그인이 안 돼요"}'

# 2. OpenAI SDK로 직접 (vLLM은 127.0.0.1:8001)
export OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=unused MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -m src.client_demo
uv run python -m src.eval_served --model flower-support-adapter-v1
uv run python -m src.bench_llm --model flower-support-adapter-v1 --concurrency 1 4 16 64

# 3. 새 버전 승인 → 반영, 롤백
uv run python -m src.promote approve && curl -s -X POST localhost:8002/reload -H "X-Reload-Token: $RELOAD_TOKEN"
uv run python -m src.promote rollback && curl -s -X POST localhost:8002/reload -H "X-Reload-Token: $RELOAD_TOKEN"

# 4. 양자화 (별도 환경, GPU)
uv run python -m src.merge outputs/<실행>/adapter                         # 8장: 병합 모델
cd quantize && uv sync && uv run python quantize.py <병합 폴더> <저장 폴더> ../data/prepared_v2/train.jsonl
```
