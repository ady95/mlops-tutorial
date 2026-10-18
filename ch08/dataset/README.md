# 8장 합성 데이터: 꽃 배달 쇼핑몰 고객 문의 3,000건

`flower-support/src/make_dataset.py`로 LLM API를 이용해 합성한 **가상의** 고객 문의입니다. 실제 고객 정보는 들어 있지 않습니다. 08-1에서 실측에 쓴 파일 그대로입니다(MD5 `acc6c6e5b2334e24265b674356ac94e4`).

| 필드 | 내용 |
|---|---|
| `id` | 0~2999 |
| `category`, `urgency`, `sentiment`, `order_id`, `product` | 문의를 쓰기 전에 먼저 정한 라벨 (정답) |
| `topic`, `style` | 문의 작성에 준 사연과 말투 |
| `inquiry`, `summary` | 생성된 문의와 한 문장 요약 |
| `check` | 다른 모델이 라벨을 모른 채 다시 분류한 결과 (검수용) |

API 키 없이 따라 하려면 이 파일을 `flower-support/data/raw/`에 복사한 뒤 DVC로 등록하세요.

```bash
cd flower-support
mkdir -p data/raw && cp ../dataset/generated.jsonl data/raw/
uv run dvc init && uv run dvc add data/raw && uv run dvc repro
```

라이선스는 저장소와 같은 MIT입니다.
