"""꽃 사진 원본 데이터를 내려받아 data/raw/ 에 푼다 (03-1).

실행: uv run python src/download.py
"""

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"

archive = Path("data/downloads/flower_photos.tgz")
raw_dir = Path("data/raw")

if not archive.exists():
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, archive)

digest = hashlib.sha256(archive.read_bytes()).hexdigest()
if digest != SHA256:
    raise SystemExit(f"체크섬 불일치: {digest} (내려받은 파일이 바뀌었거나 손상되었습니다)")

if raw_dir.exists():
    shutil.rmtree(raw_dir)
with tarfile.open(archive) as tar:
    tar.extractall("data/downloads", filter="data")
shutil.move("data/downloads/flower_photos", raw_dir)

counts = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(raw_dir.iterdir()) if d.is_dir()}
print(f"saved to {raw_dir}: {sum(counts.values())} images {counts}")
