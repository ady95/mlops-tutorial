"""원본 이미지의 품질을 검사하고, 제외할 이미지와 중복 묶음을 기록한다 (03-3).

검사 항목
  corrupt         이미지를 열 수 없음                         → 제외
  too_small       짧은 변이 min_side 미만                      → 제외
  duplicate       같은 클래스 안의 완전히 같은 파일(두 번째부터) → 제외
  label_conflict  같은 사진이 서로 다른 클래스에 들어 있음      → 모두 제외
  near_duplicate  같은 클래스 안의 거의 같은 사진               → 유지, 같은 묶음(group)으로 기록

제외 비율이나 클래스별 남은 수가 기준을 벗어나면 종료 코드 1로 멈춘다.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

params = yaml.safe_load(open("params.yaml"))["validate"]
raw_dir = Path("data/raw")
out_file = Path("data/interim/validation.json")
metrics_file = Path("reports/validation.json")


def dhash(img, size=8):
    """차이 해시: 흑백으로 줄인 뒤 이웃 픽셀의 밝기 비교를 64비트로 만든다."""
    px = img.convert("L").resize((size + 1, size)).tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            i = row * (size + 1) + col
            bits = (bits << 1) | (px[i] > px[i + 1])
    return bits


files = sorted(raw_dir.glob("*/*.jpg"))
excluded = {}                      # 상대 경로 → 제외 사유
info = {}                          # 상대 경로 → (클래스, md5, dhash)

for path in files:
    rel = path.relative_to(raw_dir).as_posix()
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            info[rel] = (path.parent.name, hashlib.md5(path.read_bytes()).hexdigest(), dhash(img))
    except Exception:
        excluded[rel] = "corrupt"
        continue
    if min(width, height) < params["min_side"]:
        excluded[rel] = "too_small"

# 완전히 같은 파일(md5)
by_md5 = defaultdict(list)
for rel, (cls, md5, _) in info.items():
    by_md5[md5].append(rel)
for rels in by_md5.values():
    if len(rels) < 2:
        continue
    if len({info[r][0] for r in rels}) > 1:
        for r in rels:
            excluded[r] = "label_conflict"
    else:
        for r in rels[1:]:
            excluded.setdefault(r, "duplicate")

# 거의 같은 사진(dHash 거리) → 묶음으로 합친다 (union-find)
parent = {rel: rel for rel in info if rel not in excluded}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


kept = sorted(parent)
for i, a in enumerate(kept):
    for b in kept[i + 1:]:
        if bin(info[a][2] ^ info[b][2]).count("1") <= params["near_dup_distance"]:
            if info[a][0] != info[b][0]:
                excluded[a] = excluded[b] = "label_conflict"
            else:
                parent[find(a)] = find(b)

groups = defaultdict(list)
for rel in kept:
    if rel not in excluded:
        groups[find(rel)].append(rel)
near_dup_groups = sorted([sorted(g) for g in groups.values() if len(g) > 1])

# 결과 집계와 판정
class_counts = Counter(info[r][0] for r in info if r not in excluded)
exclude_ratio = len(excluded) / len(files)
problems = []
if exclude_ratio > params["max_exclude_ratio"]:
    problems.append(f"제외 비율 {exclude_ratio:.4f} > {params['max_exclude_ratio']}")
for cls, n in sorted(class_counts.items()):
    if n < params["min_per_class"]:
        problems.append(f"{cls} 클래스 {n}장 < {params['min_per_class']}")

out_file.parent.mkdir(parents=True, exist_ok=True)
json.dump({"excluded": dict(sorted(excluded.items())), "near_duplicate_groups": near_dup_groups},
          open(out_file, "w"), indent=1, ensure_ascii=False)

metrics = {
    "total": len(files),
    "kept": len(files) - len(excluded),
    "excluded": dict(Counter(excluded.values())),
    "exclude_ratio": round(exclude_ratio, 4),
    "near_duplicate_groups": len(near_dup_groups),
    "class_counts": dict(sorted(class_counts.items())),
    "passed": not problems,
}
metrics_file.parent.mkdir(parents=True, exist_ok=True)
json.dump(metrics, open(metrics_file, "w"), indent=2, ensure_ascii=False)
print(json.dumps(metrics, indent=2, ensure_ascii=False))

if problems:
    print("품질 검사 실패:", "; ".join(problems), file=sys.stderr)
    sys.exit(1)
