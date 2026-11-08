"""Split the dataset into train / val / test folders (70 / 15 / 15), and pad the training
folders of small corn classes with flipped / rotated / brightened copies.

Run from the project root (and again whenever photos are added to or removed from raw/ or
not_corn/):
    ml/.venv/Scripts/python.exe ml/src/make_split.py

Makes:
    ml/dataset/split/<set>/corn/<class>/<photo>         copy of ml/dataset/raw/corn/<class>/<photo>
    ml/dataset/split/<set>/not_corn/<plant>/<photo>     copy of ml/dataset/not_corn/<plant>/<photo>
    ml/dataset/split/train/corn/<class>/aug-<kind>_<photo>   augmented copy (small classes only)
    ml/dataset/split.csv                                which set each photo is in
<set> is train, val or test.

Rules:
- Every folder is split 70/15/15 on its own. Inside a corn folder, the "insect visible" yes,
  no and not-yet-labeled photos are each split 70/15/15 too.
- Photos of one iNaturalist observation (inat_<id>.jpg, inat_<id>_p2.jpg ...), one GBIF record,
  one PlantVillage leaf (its "copy" files are re-shots or warped copies of the same leaf), or
  byte-identical files always land in the same set, so nothing is in both train and test.
- Only training photos are augmented, so val and test hold real photos only. A corn class
  with fewer than CLASS_TARGET photos gets copies until its training folder has TRAIN_TARGET.
- A photo keeps its set on later runs (read from split.csv); only new photos are placed.

raw/ and not_corn/ are never changed. Copies in split/ whose photo is gone from raw/ (or that
are no longer needed) are moved to ml/dataset/reserve/split_removed/, not deleted.
"""
import csv
import hashlib
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from features import decode_image

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
RAW_DIR = DATASET_DIR / "raw"
NOT_CORN_DIR = DATASET_DIR / "not_corn"
SPLIT_DIR = DATASET_DIR / "split"
SPLIT_FILE = DATASET_DIR / "split.csv"
LABELS_FILE = DATASET_DIR / "labels.csv"
CREDITS_FILE = DATASET_DIR / "dataset_credits.csv"
REMOVED_DIR = DATASET_DIR / "reserve" / "split_removed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASS_TARGET = 5000
TRAIN_TARGET = round(CLASS_TARGET * SHARES["train"])  # 3500: training photos per corn class after padding
SEED = 42
AUG_PREFIX = "aug-"


def _rotate15(rgb):
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.25)  # zoom in so no black corners show
    return cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


AUGMENTATIONS = {
    "hflip": lambda im: im[:, ::-1],
    "vflip": lambda im: im[::-1, :],
    "rot90": lambda im: np.rot90(im),
    "rot15": _rotate15,
    "bright": lambda im: np.clip(im.astype(np.float32) * 1.2 + 8, 0, 255).astype(np.uint8),
}


def original_name(name):
    """aug-hflip_inat_123.jpg -> inat_123.jpg (unchanged for real photos)."""
    return name.split("_", 1)[1] if name.startswith(AUG_PREFIX) else name


def list_photos():
    """{"corn/<class>/<file>" or "not_corn/<plant>/<file>": source path}"""
    photos = {}
    for base, root in ((RAW_DIR, RAW_DIR), (NOT_CORN_DIR, DATASET_DIR)):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                photos[p.relative_to(root).as_posix()] = p
    return photos


def load_labels():
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        return {r["path"].replace("\\", "/"): r["bug_visible"] for r in csv.DictReader(f)}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


class Groups:
    """Union-find over photos: photos that must stay in the same set."""

    def __init__(self, rels):
        self.parent = {r: r for r in rels}

    def find(self, r):
        while self.parent[r] != r:
            self.parent[r] = self.parent[self.parent[r]]
            r = self.parent[r]
        return r

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def plantvillage_leaf(original_name):
    """PlantVillage's own rule for photos of the same leaf (its leaf_id): the part of the
    original file name after '___', without ' copy ...' and the extension, e.g. 'r.s_hl 8315'."""
    ident = original_name.replace("_final_masked", "").split("___")[-1].split("copy")[0]
    ident = re.sub(r"\.(jpe?g|png)$", "", ident.strip(), flags=re.I)
    return " ".join(ident.lower().split())


def build_groups(photos):
    groups = Groups(photos)
    by_key = defaultdict(list)
    for rel in photos:
        m = re.match(r"(inat_\d+|gbif_\d+)", rel.rsplit("/", 1)[-1])
        if m:
            by_key[m.group(1)].append(rel)
    with open(CREDITS_FILE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rel = r["file"].replace("\\", "/")
            if r["source"].lower().startswith("plantvillage") and rel in photos:
                by_key["plantvillage:" + plantvillage_leaf(r["original_name"])].append(rel)
    with ThreadPoolExecutor(8) as pool:
        for rel, h in zip(photos, pool.map(md5, photos.values())):
            by_key["md5:" + h].append(rel)
    identical = 0
    for key, rels in by_key.items():
        identical += key.startswith("md5:") and len(rels) > 1
        for r in rels[1:]:
            groups.union(rels[0], r)
    return groups, identical


def assign(photos, groups, strata):
    """{rel: set}. Keeps sets from split.csv; places new groups so every stratum stays 70/15/15."""
    old = {}
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE, newline="", encoding="utf-8") as f:
            old = {r["path"]: r["split"] for r in csv.DictReader(f)}

    members = defaultdict(list)
    for rel in photos:
        members[groups.find(rel)].append(rel)

    group_set, counts = {}, defaultdict(Counter)
    for g, rels in members.items():
        kept = Counter(old[r] for r in rels if old.get(r) in SHARES)
        if kept:
            group_set[g] = kept.most_common(1)[0][0]
            for r in rels:
                counts[strata[r]][group_set[g]] += 1

    new = defaultdict(list)  # stratum -> groups not placed yet
    for g, rels in members.items():
        if g not in group_set:
            new[Counter(strata[r] for r in rels).most_common(1)[0][0]].append(g)

    rng = random.Random(SEED)
    for stratum in sorted(new):
        todo = sorted(new[stratum])
        rng.shuffle(todo)
        c = counts[stratum]
        for g in todo:
            size = len(members[g])
            total = sum(c.values()) + size
            s = max(SHARES, key=lambda s: SHARES[s] * total - c[s])  # set furthest below its share
            group_set[g] = s
            c[s] += size
    return {rel: group_set[groups.find(rel)] for rel in photos}


def plan_augmented(split_of):
    """{train/<folder>/aug-<kind>_<name>: (source rel, kind)} for corn folders with fewer than
    CLASS_TARGET photos (full folders are never padded, even if grouping leaves train a few short)."""
    train_by_folder, total = defaultdict(list), Counter()
    for rel, s in split_of.items():
        if not rel.startswith("not_corn/"):
            total[rel.rsplit("/", 1)[0]] += 1
            if s == "train":
                train_by_folder[rel.rsplit("/", 1)[0]].append(rel)
    plan, notes = {}, {}
    for folder, rels in sorted(train_by_folder.items()):
        n, need = len(rels), TRAIN_TARGET - len(rels)
        if need <= 0 or total[folder] >= CLASS_TARGET:
            continue
        k = min(len(AUGMENTATIONS), math.ceil(need / n))
        rels = sorted(rels)
        kinds = {r: random.Random(r).sample(sorted(AUGMENTATIONS), k) for r in rels}
        random.Random(f"{SEED}:{folder}").shuffle(rels)
        # Round-robin: every photo gets its 1st copy before any photo gets a 2nd.
        pool = [(r, kinds[r][i]) for i in range(k) for r in rels][:need]
        for r, kind in pool:
            plan[f"train/{folder}/{AUG_PREFIX}{kind}_{r.rsplit('/', 1)[1]}"] = (r, kind)
        notes[folder] = (n, len(pool))
    return plan, notes


def write_augmented(dest, src, kind):
    rgb = decode_image(np.fromfile(str(src), dtype=np.uint8).tobytes())
    bgr = cv2.cvtColor(np.ascontiguousarray(AUGMENTATIONS[kind](rgb)), cv2.COLOR_RGB2BGR)
    ext = dest.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise ValueError(f"could not encode {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(dest))


def copy_original(src, dest):
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def main():
    photos = list_photos()
    labels = load_labels()
    print(f"{len(photos)} photos in raw/ and not_corn/")

    strata = {}
    for rel in photos:
        folder = rel.rsplit("/", 1)[0]
        strata[rel] = folder if rel.startswith("not_corn/") else f"{folder}|{labels.get(rel, 'unlabeled') or 'unlabeled'}"

    print("grouping photos of the same observation / identical files ...", flush=True)
    groups, identical = build_groups(photos)
    split_of = assign(photos, groups, strata)
    aug_plan, aug_notes = plan_augmented(split_of)

    wanted = {f"{s}/{rel}" for rel, s in split_of.items()} | set(aug_plan)
    existing = {p.relative_to(SPLIT_DIR).as_posix(): p for p in SPLIT_DIR.rglob("*")
                if p.is_file()} if SPLIT_DIR.exists() else {}
    relocated = removed = 0
    for key in sorted(set(existing) - wanted):
        rel = key.split("/", 1)[1]
        dest = SPLIT_DIR / split_of[rel] / rel if rel in split_of else None
        if dest is None or dest.exists():  # photo gone from raw/, or a copy no longer needed
            dest, removed = REMOVED_DIR / key, removed + 1
        else:  # photo now belongs to another set (e.g. joined a group)
            relocated += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(existing[key]), str(dest))
    if relocated:
        print(f"moved {relocated} photos to their new set")
    if removed:
        print(f"moved {removed} files that are no longer needed to {REMOVED_DIR}")

    print("copying photos ...", flush=True)
    with ThreadPoolExecutor(8) as pool:
        copied = sum(pool.map(lambda kv: copy_original(kv[1], SPLIT_DIR / split_of[kv[0]] / kv[0]),
                              photos.items()))
    print(f"  copied {copied} (the other {len(photos) - copied} were already there)")

    todo = [(SPLIT_DIR / d, photos[r], kind) for d, (r, kind) in aug_plan.items() if not (SPLIT_DIR / d).exists()]
    print(f"writing {len(todo)} augmented copies ...", flush=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda t: write_augmented(*t), todo))

    with open(SPLIT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "group", "stratum"])
        w.writeheader()
        for rel in sorted(photos):
            w.writerow({"path": rel, "split": split_of[rel], "group": groups.find(rel), "stratum": strata[rel]})

    # Report.
    per = defaultdict(Counter)
    for rel, s in split_of.items():
        per[rel.rsplit("/", 1)[0]][s] += 1
    print(f"\n{'folder':<40}{'train':>7}{'+copies':>9}{'val':>6}{'test':>6}{'total':>7}   train/val/test %")
    for folder in sorted(per):
        c, extra = per[folder], aug_notes.get(folder, (0, 0))[1]
        n = sum(c.values())
        print(f"{folder:<40}{c['train']:>7}{extra:>9}{c['val']:>6}{c['test']:>6}{n + extra:>7}   "
              + "/".join(f"{100 * c[s] / n:.1f}" for s in SHARES))
    sets_per_group, size = defaultdict(set), Counter()
    for rel, s in split_of.items():
        sets_per_group[groups.find(rel)].add(s)
        size[groups.find(rel)] += 1
    leaks = sum(len(v) > 1 for v in sets_per_group.values())
    print(f"\n{sum(n > 1 for n in size.values())} groups of 2+ photos kept together "
          f"({identical} of them identical files); groups split across sets: {leaks}")
    print(f"wrote {SPLIT_FILE}")
    print("DONE")


if __name__ == "__main__":
    main()
