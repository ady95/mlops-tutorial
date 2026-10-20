"""도구 결과에서 모델에 보내지 않을 것을 지운다 (13-2).

운영 데이터에는 내부 주소, 서버 계정, 키가 섞여 있다. 에이전트는 외부 API로 이것을 보내므로
도구 결과가 모델에 닿기 전에 지운다. 고객 문의 원문은 11장 게이트웨이가 이미 가린 것만 쓴다.
"""

import re

RULES = [
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<내부주소>"),            # IPv4(:포트)
    (re.compile(r"/home/[^/\s\"']+"), "/home/user"),                                # 계정 이름이 든 경로
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b(\s*[=:]\s*)\S+"), r"\1\2<지움>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "<지움>"),                               # API 키 모양
    (re.compile(r"\bGPU-[0-9a-f]{8}-[0-9a-f-]{27}\b"), "<GPU-ID>"),                    # GPU UUID
    (re.compile(r"\b010-\d{3,4}-\d{4}\b"), "[전화번호]"),                              # 남아 있을지 모르는 연락처
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[이메일]"),
]


def sanitize(text: str) -> str:
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    return text
