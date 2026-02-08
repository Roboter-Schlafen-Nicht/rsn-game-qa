# build_twin_dataset.py
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

LOG_DIR = Path("F:/work/lastwarrobot/data/live_logs")
OUT_IMG = Path("F:/work/lastwarrobot/data/twin_dataset/images")
OUT_LBL = Path("F:/work/lastwarrobot/data/twin_dataset/labels")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# Use your current good HELP model
MODEL_PATH = "E:\\trainingdata\\last-war-robot\\runs\\detect\\train12\\weights\\best.pt"
HELP_CLASS_ID = 2      # BTN_HELP
CONF_MIN = 0.75         # stricter than runtime

LOG_DIR = Path("F:/work/lastwarrobot/data/live_logs")
print(LOG_DIR.exists(), len(list(LOG_DIR.glob('*.png'))))


def to_yolo_line(x1, y1, x2, y2, w, h, class_id):
    """
    Convert bounding box coordinates to a YOLO-format annotation line.
    """
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main():
    model = YOLO(MODEL_PATH)

    for img_path in sorted(LOG_DIR.glob("*.png")):
        meta_path = img_path.with_suffix(".json")
        if not meta_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        meta = json.loads(meta_path.read_text())
        det = meta.get("det")

        lines: list[str] = []

        # Option A: trust runtime detection if it’s strong enough
        if det and det.get("conf", 0.0) >= CONF_MIN:
            x1, y1, x2, y2 = det["bbox"]
            lines.append(to_yolo_line(x1, y1, x2, y2, w, h, HELP_CLASS_ID))
        else:
            # Option B: rerun YOLO to be safe
            results = model.predict(source=img, imgsz=640, conf=CONF_MIN, verbose=False)
            r = results[0]
            if r.boxes is not None:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    if int(cls.item()) != HELP_CLASS_ID:
                        continue
                    if conf.item() < CONF_MIN:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    lines.append(to_yolo_line(x1, y1, x2, y2, w, h, HELP_CLASS_ID))

        if not lines:
            continue

        out_img = OUT_IMG / img_path.name
        out_lbl = OUT_LBL / (img_path.stem + ".txt")

        cv2.imwrite(str(out_img), img)
        out_lbl.write_text("\n".join(lines))
        print("wrote", out_img.name)


if __name__ == "__main__":
    main()
