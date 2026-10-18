"""검사를 통과한 이미지를 학습·검증·테스트로 나누고 크기를 맞춘다 (03-4).

- 클래스별 비율을 유지하는 층화 분할(stratified split)
- 거의 같은 사진 묶음은 반드시 같은 세트로 보낸다 (테스트 세트 누수 방지)
- 짧은 변을 image_size로 맞춘 뒤 가운데를 정사각형으로 자른다
"""

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

excluded = set(validation["excluded"])
group_of = {}
for i, group in enumerate(validation["near_duplicate_groups"]):
    for rel in group:
        group_of[rel] = f"g{i}"

# 클래스별로 "묶음" 목록을 만든다. 묶이지 않은 사진은 혼자서 한 묶음이다.
groups_by_class = defaultdict(lambda: defaultdict(list))
for path in sorted(raw_dir.glob("*/*.jpg")):
    rel = path.relative_to(raw_dir).as_posix()
    if rel in excluded:
        continue
    groups_by_class[path.parent.name][group_of.get(rel, rel)].append(rel)

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
    split_of_group[group_of.get(rel, rel)].add(split)
leaks = sum(1 for s in split_of_group.values() if len(s) > 1)

counts = Counter(assignment.values())
per_class = {s: dict(sorted(Counter(r.split("/")[0] for r, sp in assignment.items() if sp == s).items()))
             for s in ("train", "val", "test")}
metrics = {"image_size": size, "train": counts["train"], "val": counts["val"], "test": counts["test"],
           "per_class": per_class, "groups_split_across_sets": leaks}
Path("reports").mkdir(exist_ok=True)
json.dump(metrics, open("reports/prepare.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
