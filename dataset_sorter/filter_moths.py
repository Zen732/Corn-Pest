"""Move adult-moth and off-topic photos out of a caterpillar pest folder.

Farmers photograph caterpillars and damaged corn, not adult moths, so the pest classifier
should learn from those. inat_larvae.py already filters iNaturalist photos as they arrive;
this script does the same CLIP check for photos imported from other datasets (e.g. IP102).

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/filter_moths.py --class corn/armyworm            # preview
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/filter_moths.py --class corn/armyworm --apply    # move

Preview writes a contact sheet per group to dataset_sorter/moth_preview/ to check by eye.
--apply moves "moth" photos to ml/dataset/reserve/<class>_moths/ and "other" photos (pupae,
eggs, diagrams) to ml/dataset/reserve/<class>_other/, logged in reserve/corn/moth_moves.csv.
Nothing is deleted. Caterpillar photos are labeled bug_visible=yes if not labeled yet.
iNaturalist photos (inat_*) are skipped: inat_larvae.py already checked them.
"""
import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from inat_download import LABELS_FILE, RAW_DIR, append_csv, list_images
from inat_larvae import KEEP, Judge

DATASET_DIR = RAW_DIR.parent
RESERVE_DIR = DATASET_DIR / "reserve"
MOVES_FILE = RESERVE_DIR / "corn" / "moth_moves.csv"
PREVIEW_DIR = Path(__file__).resolve().parent / "moth_preview"
BATCH = 64


def labeled_paths():
    if not LABELS_FILE.exists():
        return set()
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        return {r["path"].replace("/", "\\") for r in csv.DictReader(f)}


def contact_sheet(paths, out, cols=10, size=150, limit=100):
    paths = paths[:limit]
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    for i, p in enumerate(paths):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((size, size))
        sheet.paste(im, ((i % cols) * size, (i // cols) * size))
    sheet.save(out, quality=85)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class", dest="cls", required=True, help="e.g. corn/armyworm")
    p.add_argument("--apply", action="store_true", help="move the photos (default: preview only)")
    p.add_argument("--move", default="moth,other",
                   help="groups to move, e.g. 'moth' to keep 'other' photos after checking the preview")
    args = p.parse_args()

    files = [f for f in list_images(RAW_DIR / args.cls) if not f.name.startswith("inat_")]
    judge = Judge()
    groups = {}
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        images = [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in chunk]
        for f, g in zip(chunk, judge.groups(images)):
            groups.setdefault(g, []).append(f)
    print(f"{args.cls}: checked {len(files)} non-iNaturalist photos")
    for g in ("caterpillar", "damage", "moth", "other"):
        print(f"  {g:<12}{len(groups.get(g, [])):>6}{'  keep' if g in KEEP else '  -> reserve'}")

    if not args.apply:
        PREVIEW_DIR.mkdir(exist_ok=True)
        for g, paths in groups.items():
            contact_sheet(paths, PREVIEW_DIR / f"{args.cls.replace('/', '_')}_{g}.jpg")
        print(f"Preview only. Contact sheets: {PREVIEW_DIR}")
        return

    pest = args.cls.split("/")[1]
    moved = 0
    for g in [g for g in ("moth", "other") if g in args.move.split(",")]:
        dest = RESERVE_DIR / f"{args.cls}_{'moths' if g == 'moth' else 'other'}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in groups.get(g, []):
            shutil.move(str(f), str(dest / f.name))
            append_csv(MOVES_FILE, ["file", "class", "moved_to"],
                       {"file": f.name, "class": pest, "moved_to": str(dest.relative_to(DATASET_DIR.parent.parent))})
            moved += 1
    done = labeled_paths()
    new_labels = 0
    for f in groups.get("caterpillar", []):
        rel = str(f.relative_to(RAW_DIR))
        if rel not in done:
            append_csv(LABELS_FILE, ["path", "crop", "pest_class", "bug_visible"],
                       {"path": rel, "crop": args.cls.split("/")[0], "pest_class": pest, "bug_visible": "yes"})
            new_labels += 1
    print(f"Moved {moved} photos to {RESERVE_DIR}, labeled {new_labels} caterpillars bug_visible=yes")


if __name__ == "__main__":
    main()
