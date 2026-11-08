"""Cap each class in ml/dataset/raw/ at --target images; move the excess (not delete)
into ml/dataset/reserve/<crop>/<class>/, same folder layout, so you can pull them back
in later if a class needs more.

Usage (from the project root):
    ml/.venv/Scripts/python.exe ml/src/cap_dataset.py --dry-run   # show the plan only
    ml/.venv/Scripts/python.exe ml/src/cap_dataset.py             # move for real

When a class has more than --target images, the ones moved out first are the
flipped/rotated duplicates the dataset sorter created (`*_aug*.jpg`), since those add
the least variety. Only if that's not enough does it move original photos too, chosen
at random (seeded, so a dry run and the real run agree).
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
RAW_DIR = DATASET_DIR / "raw"
RESERVE_DIR = DATASET_DIR / "reserve"
LOG_FILE = DATASET_DIR / "cap_log.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=300, help="max images to keep per class (default 300)")
    p.add_argument("--dry-run", action="store_true", help="show what would move, move nothing")
    return p.parse_args()


def list_images(folder: Path):
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suffix, i = dest.stem, dest.suffix, 2
    while (dest.parent / f"{stem}_{i}{suffix}").exists():
        i += 1
    return dest.parent / f"{stem}_{i}{suffix}"


def main():
    args = parse_args()
    rng = random.Random(SEED)
    log_rows = []

    if not RAW_DIR.is_dir():
        raise SystemExit(f"not found: {RAW_DIR}")

    print(f"{'class':<28}{'before':>8}{'moved':>8}{'after':>8}")
    for crop_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
            images = list_images(class_dir)
            before = len(images)
            excess = before - args.target
            if excess <= 0:
                print(f"{crop_dir.name}/{class_dir.name:<20}{before:>8}{0:>8}{before:>8}")
                continue

            augmented = [f for f in images if "_aug" in f.stem]
            originals = [f for f in images if f not in set(augmented)]
            rng.shuffle(augmented)
            rng.shuffle(originals)
            to_move = (augmented + originals)[:excess]  # augmented ones leave first

            dest_dir = RESERVE_DIR / crop_dir.name / class_dir.name
            for f in to_move:
                dest = unique_destination(dest_dir / f.name)
                log_rows.append({"source": str(f.relative_to(RAW_DIR)),
                                  "destination": str(dest.relative_to(DATASET_DIR))})
                if not args.dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))

            print(f"{crop_dir.name}/{class_dir.name:<20}{before:>8}{len(to_move):>8}{before - len(to_move):>8}")

    if args.dry_run:
        print("\nDRY RUN, nothing was moved")
    elif log_rows:
        write_header = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source", "destination"])
            if write_header:
                w.writeheader()
            w.writerows(log_rows)
        print(f"\nMoved {len(log_rows)} image(s) to {RESERVE_DIR}. Log: {LOG_FILE}")
    else:
        print("\nNothing needed moving.")


if __name__ == "__main__":
    main()
