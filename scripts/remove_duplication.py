from pathlib import Path
from PIL import Image
import imagehash
import shutil

FRAMES_DIR = Path("E:\\lastwar_raw_images")      # your VLC output dir
KEEP_DIR   = Path("E:\\lastwar_deduped_images")  # cleaned set
KEEP_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 3  # max Hamming distance to still treat as duplicate

hashes = []

for img_path in sorted(FRAMES_DIR.glob("*.png")):
    img = Image.open(img_path)
    h = imagehash.phash(img)  # or dhash/average_hash
    img.close()

    is_dupe = False
    for old_h in hashes:
        if h - old_h <= THRESHOLD:   # small distance = very similar
            is_dupe = True
            break

    if not is_dupe:
        hashes.append(h)
        shutil.copy2(img_path, KEEP_DIR / img_path.name)
