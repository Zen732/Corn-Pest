"""Pick images from big datasets that look like your example photos.

Put whole datasets in  dataset_sorter/inbox/
Put example photos in  dataset_sorter/examples/<crop>/<class>/
Run (from the project root):
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/sort_images.py --dry-run
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/sort_images.py

Clear matches are COPIED to  ml/dataset/raw/<crop>/<class>/
Unsure matches are COPIED to dataset_sorter/review/<crop>/<class>/
Originals in inbox/ are never changed.
"""
import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SORTER_DIR = Path(__file__).resolve().parent
INBOX_DIR = SORTER_DIR / "inbox"
EXAMPLES_DIR = SORTER_DIR / "examples"
REVIEW_DIR = SORTER_DIR / "review"
RAW_DIR = SORTER_DIR.parent / "ml" / "dataset" / "raw"
LOG_FILE = SORTER_DIR / "sort_log.csv"
DRYRUN_LOG_FILE = SORTER_DIR / "sort_log_dryrun.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_FIELDS = ["source", "class", "score", "second_best", "action", "destination"]
DUPLICATE_DISTANCE = 4  # max perceptual-hash difference (out of 64 bits) to call two images the same
LOAD_SIZE = 512


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="show what would happen, copy nothing")
    p.add_argument("--high", type=float, default=0.85, help="score needed for a clear match (default 0.85)")
    p.add_argument("--low", type=float, default=0.75, help="score needed to go to review (default 0.75)")
    p.add_argument("--margin", type=float, default=0.03,
                   help="best class must beat the second best by this much (default 0.03)")
    p.add_argument("--target", type=int, default=300,
                   help="images per class at which duplicates start being skipped instead of "
                        "augmented (default 300)")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="stop adding to a class once it has this many images (default 0 = no limit)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4, help="image loading workers (use 0 if you get errors)")
    return p.parse_args()


# ---------------------------------------------------------------- helpers

def list_images(folder: Path):
    return sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path):
    """Open an image as RGB. JPEGs are decoded at reduced size to save time."""
    from PIL import Image, ImageOps
    img = Image.open(path)
    if img.format == "JPEG":
        img.draft("RGB", (LOAD_SIZE, LOAD_SIZE))
    return ImageOps.exif_transpose(img).convert("RGB")


def phash_int(img) -> int:
    import imagehash
    return int(str(imagehash.phash(img)), 16)


def is_duplicate(h: int, known) -> bool:
    return any(bin(h ^ k).count("1") <= DUPLICATE_DISTANCE for k in known)


def safe_name(source: Path) -> str:
    """dataset_originalname_shorthash.ext, so same file names from different datasets don't clash."""
    rel = source.relative_to(INBOX_DIR)
    dataset = re.sub(r"[^A-Za-z0-9]+", "-", rel.parts[0])[:30] if len(rel.parts) > 1 else "inbox"
    stem = re.sub(r"[^A-Za-z0-9]+", "-", source.stem)[:40]
    short = hashlib.sha1(str(rel).encode()).hexdigest()[:6]
    return f"{dataset}_{stem}_{short}{source.suffix.lower()}"


def augment(img, rng: random.Random):
    """Return a visibly different version: flip / 90-degree turn, or a small rotation with a crop."""
    from PIL import Image
    ops = [
        ("flipLR", lambda i: i.transpose(Image.FLIP_LEFT_RIGHT)),
        ("flipTB", lambda i: i.transpose(Image.FLIP_TOP_BOTTOM)),
        ("rot90", lambda i: i.transpose(Image.ROTATE_90)),
        ("rot180", lambda i: i.transpose(Image.ROTATE_180)),
        ("rot270", lambda i: i.transpose(Image.ROTATE_270)),
        ("rotsmall", None),
        ("flip+rotsmall", None),
    ]
    name, fn = rng.choice(ops)
    if fn is not None:
        return name, fn(img)
    if name == "flip+rotsmall":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.choice([-1, 1]) * rng.uniform(8, 20)
    w, h = img.size
    rotated = img.rotate(angle, resample=Image.BICUBIC)
    cw, ch = int(w * 0.75), int(h * 0.75)  # crop away the black corners
    left, top = (w - cw) // 2, (h - ch) // 2
    out = rotated.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)
    return f"{name}{angle:+.0f}", out


# ---------------------------------------------------------------- embedding

def _collate(batch):
    return batch


class ImageData(Dataset):
    """Module-level so Windows worker processes can pickle it."""

    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = load_image(self.paths[i])
            return i, self.preprocess(img), phash_int(img)
        except Exception:
            return i, None, None


def embed_images(paths, model, preprocess, device, batch_size, workers):
    """Return (embeddings list of float32 arrays, phashes list[int|None]) in the order of `paths`."""
    loader = DataLoader(ImageData(paths, preprocess), batch_size=batch_size,
                        num_workers=workers, collate_fn=_collate)
    embeddings = [None] * len(paths)
    hashes = [None] * len(paths)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding", unit="batch"):
            good = [b for b in batch if b[1] is not None]
            if not good:
                continue
            x = torch.stack([b[1] for b in good]).to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                feats = model.encode_image(x)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()
            for (i, _, h), f in zip(good, feats):
                embeddings[i] = f
                hashes[i] = h
    return embeddings, hashes


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    if args.low > args.high:
        sys.exit("--low must not be greater than --high")

    # ---- classes and examples
    example_paths, example_class = [], []
    for crop_dir in sorted(p for p in EXAMPLES_DIR.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
            imgs = list_images(class_dir)
            if not imgs:
                print(f"  (no examples for {crop_dir.name}/{class_dir.name}, that class is skipped)")
            for f in imgs:
                example_paths.append(f)
                example_class.append(f"{crop_dir.name}/{class_dir.name}")
    if not example_paths:
        sys.exit("No example images found. Put 3-10 photos per class in dataset_sorter/examples/<crop>/<class>/")

    classes = sorted(set(example_class))
    class_index = {c: [i for i, k in enumerate(example_class) if k == c] for c in classes}
    print("Classes with examples:")
    for c in classes:
        print(f"  {c:<28} {len(class_index[c])} example(s)")

    # ---- inbox images (skip files handled in earlier runs)
    processed = set()
    if LOG_FILE.exists() and not args.dry_run:
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            processed = {r["source"] for r in csv.DictReader(f)}
    inbox_paths = [p for p in list_images(INBOX_DIR) if str(p.relative_to(INBOX_DIR)) not in processed]
    if not inbox_paths:
        sys.exit("Nothing new in dataset_sorter/inbox/")
    print(f"\n{len(inbox_paths)} new image(s) in inbox")

    # ---- model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-B/32 on {device} (first run downloads the model, about 350 MB)")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()

    print("\nEmbedding examples")
    ex_emb, _ = embed_images(example_paths, model, preprocess, device, args.batch_size, 0)
    ex_ok = [i for i, e in enumerate(ex_emb) if e is not None]
    if len(ex_ok) < len(ex_emb):
        print(f"  warning: {len(ex_emb) - len(ex_ok)} example image(s) could not be read")
    ex_matrix = np.stack([ex_emb[i] for i in ex_ok])
    ok_index = {c: [j for j, i in enumerate(ex_ok) if example_class[i] == c] for c in classes}
    classes = [c for c in classes if ok_index[c]]

    print("\nEmbedding inbox")
    emb, hashes = embed_images(inbox_paths, model, preprocess, device, args.batch_size, args.workers)

    # ---- score every inbox image against every class
    rows = []  # (path, best_class, best_score, second_score, hash)
    log = []
    unreadable = 0
    for path, e, h in zip(inbox_paths, emb, hashes):
        if e is None:
            unreadable += 1
            log.append({"source": str(path.relative_to(INBOX_DIR)), "class": "", "score": "",
                        "second_best": "", "action": "unreadable", "destination": ""})
            continue
        sims = ex_matrix @ e
        scores = sorted(((float(sims[ok_index[c]].max()), c) for c in classes), reverse=True)
        second = scores[1][0] if len(scores) > 1 else 0.0
        rows.append((path, scores[0][1], scores[0][0], second, h))

    # ---- existing raw images: counts and hashes for duplicate detection
    counts = Counter()
    known_hashes = defaultdict(list)
    for c in classes:
        folder = RAW_DIR / c
        if folder.is_dir():
            for f in list_images(folder):
                counts[c] += 1
                try:
                    known_hashes[c].append(phash_int(load_image(f)))
                except Exception:
                    pass

    # ---- decide and copy
    summary = defaultdict(Counter)
    rng = random.Random(42)
    rows.sort(key=lambda r: -r[2])  # best matches first
    for path, cls, score, second, h in rows:
        rel = str(path.relative_to(INBOX_DIR))
        entry = {"source": rel, "class": cls, "score": f"{score:.3f}", "second_best": f"{second:.3f}"}
        clear = score >= args.high and (score - second) >= args.margin

        if clear:
            if args.max_per_class and counts[cls] >= args.max_per_class:
                action, dest = "skipped_class_full", ""
            elif is_duplicate(h, known_hashes[cls]):
                if counts[cls] >= args.target:
                    action, dest = "skipped_duplicate", ""
                else:
                    action = "augmented_duplicate"
                    dest_path = RAW_DIR / cls / (Path(safe_name(path)).stem + "_aug.jpg")
                    if not args.dry_run:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        _, aug = augment(load_image(path), rng)
                        k = 1
                        while dest_path.exists():  # same file matched again with another transform
                            k += 1
                            dest_path = dest_path.with_name(f"{Path(safe_name(path)).stem}_aug{k}.jpg")
                        aug.save(dest_path, quality=95)
                        known_hashes[cls].append(phash_int(aug))
                    counts[cls] += 1
                    dest = str(dest_path.relative_to(SORTER_DIR.parent))
            else:
                action = "copied"
                dest_path = RAW_DIR / cls / safe_name(path)
                if not args.dry_run:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest_path)
                known_hashes[cls].append(h)
                counts[cls] += 1
                dest = str(dest_path.relative_to(SORTER_DIR.parent))
        elif score >= args.low:
            action = "review"
            dest_path = REVIEW_DIR / cls / safe_name(path)
            if not args.dry_run:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest_path)
            dest = str(dest_path.relative_to(SORTER_DIR.parent))
        else:
            action, dest = "no_match", ""
            entry["class"] = ""

        entry.update(action=action, destination=dest)
        log.append(entry)
        summary[cls][action] += 1

    # ---- write log
    if args.dry_run:
        out, mode = DRYRUN_LOG_FILE, "w"
    else:
        out, mode = LOG_FILE, "a"
    write_header = mode == "w" or not out.exists()
    with open(out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(log)

    # ---- report
    print("\n" + ("DRY RUN, nothing was copied\n" if args.dry_run else ""))
    print(f"{'class':<28}{'copied':>8}{'augmented':>11}{'dup skipped':>13}{'review':>8}")
    for c in classes:
        s = summary[c]
        print(f"{c:<28}{s['copied']:>8}{s['augmented_duplicate']:>11}"
              f"{s['skipped_duplicate'] + s['skipped_class_full']:>13}{s['review']:>8}")
    no_match = sum(1 for r in log if r["action"] == "no_match")
    print(f"\nNo match: {no_match}   Unreadable: {unreadable}")
    print(f"Log: {out}")
    print("Tip: open the log and sort by score to see if --high / --low fit your data.")


if __name__ == "__main__":
    main()
