"""지금 나뉜 세트를 data/split.csv로 고정한다 (12-5). 데이터를 더하기 전에 한 번 실행하고 커밋한다.

    uv run python -m src.freeze_split                 # 지금 data/prepared 기준

이후 prepare.py는 이 파일에 있는 사진을 그 세트 그대로 두고, 새 사진만 배정한다.
"""

import csv
import sys
from pathlib import Path

prepared = Path(sys.argv[1] if len(sys.argv) > 1 else "data/prepared")
rows = sorted((p.relative_to(prepared / s).as_posix(), s)
              for s in ("train", "val", "test") for p in (prepared / s).glob("*/*.jpg")
              if not p.name.startswith("night_"))
with open("data/split.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["path", "split"])
    writer.writerows(rows)
print(f"data/split.csv: {len(rows)}장 고정")
