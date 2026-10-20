"""끝난 코딩 작업 폴더를 다시 채점한다 (인수 테스트를 고친 뒤 등)."""

import json
import shutil
import sys
from pathlib import Path

from codegen import lost_comments, pytest

out = Path(sys.argv[1])
for ws in sorted(p for p in out.iterdir() if p.is_dir()):
    task = ws.name.split("-")[0]
    shutil.copy(Path("cases/codegen") / task / "test_acceptance.py", ws / "tests" / "test_acceptance.py")
    acc = pytest(ws, "tests/test_acceptance.py")
    print(json.dumps({"ws": ws.name, "acceptance": acc, "lost_comments": lost_comments(ws)}, ensure_ascii=False))
