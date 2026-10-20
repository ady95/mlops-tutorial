"""조치 제안 도구 (13-5). 에이전트가 '바꾸는' 일을 하는 유일한 길은 승인 요청 이슈를 여는 것이다.

실행 권한은 에이전트에게 없다. 사람이 이슈에 /approve를 달면 비공개 저장소의 approve 워크플로가
src/actions.py로 이슈 속 JSON을 검사하고, 허용 목록(ACTIONS)에 있는 조치만 실행한다.
에이전트의 GitHub 토큰은 이슈 쓰기 권한만 있으면 된다.

환경변수: GITHUB_TOKEN(이슈 쓰기), GITHUB_REPO(예: owner/mlops-pipeline)
"""

import json
import os

import httpx

from agent.core import tool

# 실행기(src/actions.py)와 같은 목록. 여기서 먼저 걸러 에이전트에게 바로 알려 준다.
ACTIONS = {"deploy": ["version"], "restore": ["version"], "silence": ["alertname", "hours"], "retrain": []}


@tool("사람에게 운영 조치를 제안한다(승인 요청 이슈). 직접 실행하지 않는다. "
      "action: deploy(새 버전을 champion으로, 배포 판정 재확인 포함) | restore(승인을 거쳐 운영된 적 있는 버전으로 되돌리기) | "
      "silence(알림 묵음, 최대 24시간) | retrain(학습 파이프라인 실행)",
      writes=True,
      action={"type": "string", "enum": list(ACTIONS)},
      args={"type": "object", "description": "deploy·restore: {version}, silence: {alertname, hours}, retrain: {}"},
      title={"type": "string"}, reason={"type": "string", "description": "왜 이 조치인가"},
      evidence={"type": "string", "description": "근거가 된 지표·시각·값"},
      risk={"type": "string", "description": "이 조치가 틀렸을 때 생기는 일과 되돌리는 법"})
def propose_action(action: str, args: dict, title: str, reason: str, evidence: str, risk: str) -> str:
    if action not in ACTIONS or sorted(args) != sorted(ACTIONS[action]):
        return f"제안할 수 없음: action={action}, args={sorted(args)}. 가능한 형식: {ACTIONS}"
    plan = json.dumps({"action": action, **{k: str(v) for k, v in args.items()}}, ensure_ascii=False)
    body = (f"## 제안\n{reason}\n\n## 근거\n{evidence}\n\n## 위험과 되돌리기\n{risk}\n\n"
            f"## 승인하면 실행할 조치\n```json\n{plan}\n```\n\n"
            "승인하려면 이 이슈에 `/approve` 댓글을 다세요. 승인하지 않으면 아무 일도 일어나지 않습니다.\n"
            "(운영 보조 에이전트가 작성)")
    r = httpx.post(f"https://api.github.com/repos/{os.environ['GITHUB_REPO']}/issues", timeout=30,
                   headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"},
                   json={"title": f"[에이전트 제안] {title}", "body": body, "labels": ["agent-proposal"]})
    r.raise_for_status()
    return f"승인 요청을 열었음: #{r.json()['number']} — 사람이 승인하기 전에는 실행되지 않음"


ACTION_TOOLS = [propose_action]
