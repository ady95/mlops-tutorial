"""답에 나온 숫자가 도구 결과에 실제로 있었는가 (13-3).

에이전트의 분석에서 가장 위험한 실수는 그럴듯한 숫자를 지어내는 것이다. 답의 소수(0.9096, 8.6 등)를
모두 뽑아, 그 실행에서 도구가 돌려준 결과 안에 같은 값(답의 자릿수로 반올림해서)이 있었는지 본다.
"""

import re

NUM = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")


def numbers(text: str) -> list[str]:
    return NUM.findall(text or "")


def check(answer: str, messages: list[dict]) -> dict:
    seen = set()
    for m in messages:
        if m.get("role") == "tool":
            for n in numbers(m["content"]):
                v = float(n)
                for d in range(1, 5):             # 도구 값을 1~4자리로 반올림한 모든 표기
                    seen.add(f"{round(v, d):.{d}f}")
    found = numbers(answer)
    missing = [n for n in found if f"{float(n):.{len(n.split('.')[1])}f}" not in seen]
    return {"numbers": len(found), "unsupported": missing}
