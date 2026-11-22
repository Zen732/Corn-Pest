"""Extract features.py vectors for every photo in ml/dataset/split/ (made by make_split.py)
and cache them, so train.py doesn't have to re-read every photo on each run.

Run after make_split.py, and again whenever the split or the labels change (from the project root):
    ml/.venv/Scripts/python.exe ml/src/extract_all.py
    ml/.venv/Scripts/python.exe ml/src/extract_all.py --reuse   (only read photos that are new)

--reuse copies the saved vector of every real photo already in features.npy (same set, same
path) and reads only the rest; augmented copies are always read again. Use it only when
features.py has not changed since features.npy was written (the vectors are deterministic).
A few reused photos are read again and compared, to be sure.

Corn photos need a bug_visible label in ml/dataset/labels.csv (ml/src/label_bug_visible.py);
unlabeled ones are skipped. Augmented copies (aug-<kind>_<photo>, only in split/train/) take
their original photo's label. "Not corn" photos need no label.

It also records what the photo's border looks like (white table, black background, or a
natural scene), so train.py can check whether a model is just learning backgrounds.

Writes:
    ml/results/features.npy       float32 array, shape (N, FEATURE_LENGTH)
    ml/results/features_meta.csv  one row per vector: path (the original photo, e.g.
                                  corn/armyworm/inat_123.jpg), split (train/val/test), crop,
                                  pest_class, bug_visible, aug ("" for a real photo, else its
                                  kind, e.g. hflip), background, border_v, border_s
"""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed

from features import decode_image, extract_features
from make_split import AUG_PREFIX, IMAGE_EXTENSIONS, LABELS_FILE, SHARES, SPLIT_DIR, original_name

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
NOT_CORN = "not_corn"

META_FIELDS = ["path", "split", "crop", "pest_class", "bug_visible", "aug", "background", "border_v", "border_s"]


def border_stats(rgb):
    """Mean brightness and saturation of the outer ring of the photo, and a rough type."""
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    ring = np.ones((256, 256), bool)
    ring[12:-12, 12:-12] = False
    v, s = float(hsv[..., 2][ring].mean()), float(hsv[..., 1][ring].mean())
    kind = "white" if v > 0.75 and s < 0.18 else "black" if v < 0.18 else "natural"
    return kind, v, s


def load_labels():
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        return {row["path"].replace("\\", "/"): row for row in csv.DictReader(f)}


def _extract(path_str):
    """Features + border stats of one photo. Runs in a worker process."""
    try:
        rgb = decode_image(np.fromfile(path_str, dtype=np.uint8).tobytes())
        return extract_features(rgb), border_stats(rgb)
    except Exception as e:
        print(f"  FAILED {path_str}: {e}")
        return None, None


def list_split(labels):
    """(file, meta row) for every photo in split/ that can be used."""
    files, rows, skipped, mismatched = [], [], Counter(), 0
    for p in sorted(SPLIT_DIR.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            continue
        parts = p.relative_to(SPLIT_DIR).parts
        if len(parts) != 4 or parts[0] not in SHARES:
            print(f"  ignoring {p} (expected split/<set>/<crop>/<folder>/<photo>)")
            continue
        split, crop, folder, name = parts
        aug = name[len(AUG_PREFIX):].split("_", 1)[0] if name.startswith(AUG_PREFIX) else ""
        path = f"{crop}/{folder}/{original_name(name)}"
        if crop == NOT_CORN:
            pest_class, bug = NOT_CORN, ""
        else:
            label = labels.get(path)
            if label is None:
                skipped[split] += 1
                continue
            pest_class, bug = folder, label["bug_visible"]
            mismatched += label["pest_class"] != folder
        files.append(p)
        rows.append({"path": path, "split": split, "crop": crop, "pest_class": pest_class,
                     "bug_visible": bug, "aug": aug})
    if skipped:
        print(f"  skipping {sum(skipped.values())} unlabeled photo(s)/copies "
              f"({', '.join(f'{s} {n}' for s, n in skipped.items())}) - run ml/src/label_bug_visible.py first")
    if mismatched:
        print(f"  note: {mismatched} photo(s) sit in a different folder than labels.csv says; using the folder")
    return files, rows


def saved_results():
    """{(split, path): (vector, border stats)} for real photos in the saved features.npy."""
    if not (RESULTS_DIR / "features.npy").exists():
        return {}
    X = np.load(RESULTS_DIR / "features.npy")
    with open(RESULTS_DIR / "features_meta.csv", newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    keys = Counter((m["split"], m["path"]) for m in meta if not m["aug"])
    return {(m["split"], m["path"]): (X[i], (m["background"], float(m["border_v"]), float(m["border_s"])))
            for i, m in enumerate(meta) if not m["aug"] and keys[(m["split"], m["path"])] == 1}


def main():
    ap = argparse.ArgumentParser(description="Extract features for every photo in split/.")
    ap.add_argument("--reuse", action="store_true", help="reuse saved vectors of unchanged real photos")
    reuse = ap.parse_args().reuse
    if not SPLIT_DIR.is_dir():
        raise SystemExit(f"{SPLIT_DIR} not found - run ml/src/make_split.py first")
    files, rows = list_split(load_labels())
    counts = Counter((r["split"], "copy" if r["aug"] else "real") for r in rows)
    print(f"{len(files)} photos: " + ", ".join(
        f"{s} {counts[(s, 'real')]}" + (f" (+{counts[(s, 'copy')]} copies)" if counts[(s, 'copy')] else "")
        for s in SHARES))

    saved = saved_results() if reuse else {}
    results = [None if r["aug"] else saved.get((r["split"], r["path"])) for r in rows]
    todo = [i for i, res in enumerate(results) if res is None]
    print(f"reusing {len(rows) - len(todo)} saved vectors, reading {len(todo)} photos")
    for i, res in zip(todo, Parallel(n_jobs=-1, verbose=5)(delayed(_extract)(str(files[i])) for i in todo)):
        results[i] = res
    if reuse and len(todo) < len(rows):  # spot-check: a reused vector must equal a fresh one
        sample = random.Random(0).sample([i for i in range(len(rows)) if i not in set(todo)], 25)
        for i in sample:
            fresh, _ = _extract(str(files[i]))
            if fresh is None or not np.allclose(fresh, results[i][0], atol=1e-5):
                raise SystemExit(f"saved vector differs for {rows[i]['path']}: features changed, run without --reuse")
        print("spot-check: 25 reused vectors match a fresh read")

    X, meta = [], []
    for row, (vec, border) in zip(rows, results):
        if vec is None:
            continue
        kind, v, s = border
        X.append(vec)
        meta.append({**row, "background": kind, "border_v": f"{v:.4f}", "border_s": f"{s:.4f}"})
    if not X:
        raise SystemExit("No features extracted - check the errors above.")

    X = np.stack(X).astype(np.float32)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "features.npy", X)
    with open(RESULTS_DIR / "features_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(meta)
    n_real = sum(1 for m in meta if not m["aug"])
    print(f"Saved {X.shape[0]} feature vectors ({n_real} photos + {X.shape[0] - n_real} augmented copies), "
          f"length {X.shape[1]}, to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
