"""검사를 통과한 이미지를 학습·검증·테스트로 나누고 크기를 맞춘다 (03-4, 12-5).

- 처음 나눌 때: 클래스별 비율을 유지하는 층화 분할(stratified split), 거의 같은 사진 묶음은 같은 세트로 (03-4)
- 데이터를 더할 때 (12-5): data/split.csv에 있는 사진은 그 세트 그대로 둔다. 원본을 바꿔 만든 사본(night_)은
  원본의 세트를, 새 사진은 묶음 이름의 해시로 세트를 정한다. 처음처럼 다시 섞어 나누면 지금 champion이
  학습한 사진이 새 테스트 세트에 들어가 비교가 불공정해진다 (실측: 새 테스트 원본 548장 중 381장)
- 짧은 변을 image_size로 맞춘 뒤 가운데를 정사각형으로 자른다
"""

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps

params = yaml.safe_load(open("params.yaml"))["prepare"]
raw_dir = Path("data/raw")
out_dir = Path("data/prepared")
validation = json.load(open("data/interim/validation.json"))

split_file = Path("data/split.csv")
frozen = {r["path"]: r["split"] for r in csv.DictReader(open(split_file, encoding="utf-8"))} if split_file.exists() else {}

excluded = set(validation["excluded"])
group_of = {}
for i, group in enumerate(validation["near_duplicate_groups"]):
    for rel in group:
        group_of[rel] = f"g{i}"


def is_copy(rel: str) -> bool:
    return rel.split("/")[1].startswith("night_")


def original(rel: str) -> str:
    """night_ 사본이면 원본의 경로, 아니면 그대로."""
    cls, name = rel.split("/")
    return f"{cls}/{name.removeprefix('night_')}"


def group_key(rel: str) -> str:
    """같은 세트로 보내야 하는 사진끼리 같은 값. night_ 사본은 원본의 묶음을 따른다 (12-5)."""
    return group_of.get(original(rel), original(rel))


def initial_split(groups_by_class: dict) -> dict:
    """처음 나누는 경우 (03-4): 클래스별로 묶음을 섞어 비율대로 채운다."""
    rng = random.Random(params["seed"])
    assignment = {}
    for cls in sorted(groups_by_class):
        groups = [groups_by_class[cls][k] for k in sorted(groups_by_class[cls])]
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        n_test, n_val = round(total * params["test_ratio"]), round(total * params["val_ratio"])
        filled = Counter()
        for group in groups:
            split = "test" if filled["test"] < n_test else "val" if filled["val"] < n_val else "train"
            filled[split] += len(group)
            for rel in group:
                assignment[rel] = split
    return assignment


def hash_split(key: str) -> str:
    """새 사진의 세트를 이름의 해시로 정한다. 데이터가 더 늘어도 바뀌지 않는다."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < params["test_ratio"] else "val" if h < params["test_ratio"] + params["val_ratio"] else "train"


def incremental_split(groups_by_class: dict) -> dict:
    """데이터를 더하는 경우 (12-5): 고정한 사진은 그대로, 사본은 원본을 따라, 새 사진은 해시로."""
    assignment = {}
    for groups in groups_by_class.values():
        for group in groups.values():
            known = [frozen[original(r)] for r in group if original(r) in frozen]
            fallback = known[0] if known else hash_split(min(original(r) for r in group))
            for rel in group:
                assignment[rel] = frozen.get(original(rel), fallback)
    return assignment


# 클래스별로 "묶음" 목록을 만든다. 묶이지 않은 사진은 혼자서 한 묶음이다.
groups_by_class = defaultdict(lambda: defaultdict(list))
excluded_copies = 0
for path in sorted(raw_dir.glob("*/*.jpg")):
    rel = path.relative_to(raw_dir).as_posix()
    if rel in excluded:
        continue
    if is_copy(rel) and original(rel) in excluded:         # 제외된 원본의 사본도 뺀다 (라벨 충돌 사진이 사본으로 되살아나지 않게)
        excluded_copies += 1
        continue
    groups_by_class[path.parent.name][group_key(rel)].append(rel)

assignment = incremental_split(groups_by_class) if frozen else initial_split(groups_by_class)

if out_dir.exists():
    shutil.rmtree(out_dir)
size = params["image_size"]
for rel, split in sorted(assignment.items()):
    dst = out_dir / split / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw_dir / rel) as img:
        ImageOps.fit(img.convert("RGB"), (size, size), Image.Resampling.BICUBIC).save(dst, quality=95)

# 출처 표기(CC-BY) 의무를 지키기 위해 라이선스 파일을 함께 둔다
shutil.copy(raw_dir / "LICENSE.txt", out_dir / "LICENSE.txt")

split_of_group = defaultdict(set)
for rel, split in assignment.items():
    split_of_group[group_key(rel)].add(split)
leaks = sum(1 for s in split_of_group.values() if len(s) > 1)

counts = Counter(assignment.values())
per_class = {s: dict(sorted(Counter(r.split("/")[0] for r, sp in assignment.items() if sp == s).items()))
             for s in ("train", "val", "test")}
night = Counter(sp for r, sp in assignment.items() if is_copy(r))
metrics = {"image_size": size, "train": counts["train"], "val": counts["val"], "test": counts["test"],
           "night": {s: night[s] for s in ("train", "val", "test")},
           "new_originals": sum(1 for r in assignment if not is_copy(r) and r not in frozen) if frozen else 0,
           "per_class": per_class, "groups_split_across_sets": leaks, "excluded_copies": excluded_copies}
Path("reports").mkdir(exist_ok=True)
json.dump(metrics, open("reports/prepare.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
