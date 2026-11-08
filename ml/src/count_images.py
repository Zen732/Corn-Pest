"""Count images per class in the dataset and warn about small classes.

Usage (from the project root):
    python ml/src/count_images.py                 # counts ml/dataset/raw
    python ml/src/count_images.py test_local      # counts ml/dataset/test_local
"""
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_RECOMMENDED = 100


def count_images(folder: Path) -> int:
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "raw"
    root = DATASET_DIR / split
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")

    total = 0
    for crop_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        print(f"\n{crop_dir.name.upper()}")
        for class_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
            n = count_images(class_dir)
            total += n
            flag = "" if n >= MIN_RECOMMENDED or split != "raw" else f"  <- need {MIN_RECOMMENDED - n} more"
            print(f"  {class_dir.name:<20} {n:>5}{flag}")
    print(f"\nTotal images in {split}: {total}")


if __name__ == "__main__":
    main()
