"""Interactive tool: for each photo, is a bug/insect visible, or just damage?

Needed to train the bug-visible detector (see docs/classification_flow.md).
`healthy/` photos are auto-labeled and skipped - there's nothing to judge there.

Run from the project root:
    ml/.venv/Scripts/python.exe ml/src/label_bug_visible.py

Keys while the image window is focused:
    b = bug/insect is visible in the photo
    d = damage only, no bug visible
    s = skip (ask again next run)
    u = undo the previous label
    q = quit and save

Progress is saved to ml/dataset/labels.csv after every key press, so you can stop
and resume any time. Already-labeled photos (from this or an earlier run) are
skipped automatically.
"""
import argparse
import csv
from pathlib import Path

import cv2

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
RAW_DIR = DATASET_DIR / "raw"
LABELS_FILE = DATASET_DIR / "labels.csv"
FIELDS = ["path", "crop", "pest_class", "bug_visible"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DISPLAY = 900  # longest side, in pixels, when showing the photo


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", help="only label this crop folder (e.g. corn)")
    return p.parse_args()


def load_labels():
    if not LABELS_FILE.exists():
        return {}
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def append_label(row):
    write_header = not LABELS_FILE.exists()
    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def rewrite_labels(rows):
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows.values())


def show(path, crop, pest_class, done, total):
    img = cv2.imdecode(__import__("numpy").fromfile(str(path), dtype="uint8"), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = MAX_DISPLAY / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    banner = img.copy()
    cv2.rectangle(banner, (0, 0), (img.shape[1], 34), (30, 30, 30), -1)
    text = f"{crop}/{pest_class}  [{done}/{total}]   b=bug  d=damage-only  s=skip  u=undo  q=quit"
    cv2.putText(banner, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("label_bug_visible", banner)
    return cv2.waitKeyEx(0)


def main():
    args = parse_args()
    labels = load_labels()
    history = []  # paths labeled this session, for undo

    crops = [args.crop] if args.crop else sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    todo = []
    for crop in crops:
        crop_dir = RAW_DIR / crop
        if not crop_dir.is_dir():
            print(f"no such crop folder: {crop_dir}")
            continue
        for class_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
            images = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            if class_dir.name == "healthy":
                for f in images:
                    rel = str(f.relative_to(RAW_DIR))
                    if rel not in labels:
                        row = {"path": rel, "crop": crop, "pest_class": "healthy", "bug_visible": "no"}
                        labels[rel] = row
                        append_label(row)
                continue
            for f in images:
                rel = str(f.relative_to(RAW_DIR))
                if rel not in labels:
                    todo.append((f, rel, crop, class_dir.name))

    print(f"{len(todo)} photo(s) to label (healthy photos auto-labeled and excluded)")
    if not todo:
        return

    i = 0
    while i < len(todo):
        path, rel, crop, pest_class = todo[i]
        key = show(path, crop, pest_class, i + 1, len(todo))
        ch = chr(key & 0xFF).lower() if key not in (-1, None) else ""

        if ch == "q":
            break
        elif ch == "b":
            row = {"path": rel, "crop": crop, "pest_class": pest_class, "bug_visible": "yes"}
            labels[rel] = row
            append_label(row)
            history.append(rel)
            i += 1
        elif ch == "d":
            row = {"path": rel, "crop": crop, "pest_class": pest_class, "bug_visible": "no"}
            labels[rel] = row
            append_label(row)
            history.append(rel)
            i += 1
        elif ch == "s":
            i += 1
        elif ch == "u" and history:
            last = history.pop()
            labels.pop(last, None)
            rewrite_labels(labels)
            i = max(0, i - 1)
        # any other key: redo this image

    cv2.destroyAllWindows()
    done = sum(1 for _, rel, _, _ in todo if rel in labels)
    print(f"Labeled {done}/{len(todo)} this run. Re-run any time to continue.")


if __name__ == "__main__":
    main()
