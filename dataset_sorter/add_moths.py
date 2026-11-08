"""Fill a caterpillar pest class with adult moth photos when no more caterpillar photos exist.

Order: (1) moths moved to ml/dataset/reserve/<class>_moths/ earlier are moved back (drawings
skipped), (2) iNaturalist observations that inat_larvae.py rejected as "moth" are downloaded.
Every moth photo is labeled bug_visible=yes. Moves are logged in reserve/corn/moth_moves.csv.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/add_moths.py --class corn/common_cutworm --target 5000

--swap N first moves N caterpillar/damage photos out to reserve/<class>_swapped_out/ (least
reliable first: uncredited first-download photos, then unconfirmed "needs ID" iNaturalist
photos) so a full class can get the same share of moths as the others:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/add_moths.py --class corn/fall_armyworm --target 5000 --swap 2000
"""
import argparse
import csv
import random
import shutil
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image, ImageOps

from inat_download import (API, CREDIT_FIELDS, CREDITS_FILE, LABELS_FILE, RAW_DIR, append_csv, fetch_photo,
                           first_open_photo, get_json, is_duplicate, list_images, log, phash)
from inat_larvae import REJECTED_FILE, SPECIES, Judge

DATASET_DIR = RAW_DIR.parent
RESERVE = DATASET_DIR / "reserve"
MOVES_FILE = RESERVE / "corn" / "moth_moves.csv"
DRAWING = ["a black and white drawing of an insect", "a scientific illustration", "a diagram with text"]
PHOTO = ["a photo of a moth", "a photo of an insect"]


def labeled():
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        return {r["path"].replace("/", "\\") for r in csv.DictReader(f)}


def label_yes(rel, crop, pest, done):
    if rel not in done:
        append_csv(LABELS_FILE, ["path", "crop", "pest_class", "bug_visible"],
                   {"path": rel, "crop": crop, "pest_class": pest, "bug_visible": "yes"})
        done.add(rel)


_text = {}


def is_drawing(judge, img):
    """CLIP: is this a drawing/diagram rather than a photo?"""
    with torch.no_grad():
        if "t" not in _text:
            tok = __import__("open_clip").get_tokenizer("ViT-B-32")
            t = judge.model.encode_text(tok(DRAWING + PHOTO).to(judge.device)).float()
            _text["t"] = torch.nn.functional.normalize(t, dim=-1)
        e = torch.nn.functional.normalize(judge.model.encode_image(judge.pre(img)[None].to(judge.device)).float(), dim=-1)
        p = (judge.scale * e @ _text["t"].T).softmax(-1)[0].cpu().numpy()
    return p[:len(DRAWING)].sum() > 0.5


def swap_out(folder, cls, pest, n):
    """Move n least-reliable non-moth photos to reserve: first-download, then needs-ID iNaturalist."""
    with open(CREDITS_FILE, newline="", encoding="utf-8") as fh:
        species = {r["file"].replace("/", "\\").rsplit("\\", 1)[-1]: r["species"] for r in csv.DictReader(fh)
                   if r["file"].replace("/", "\\").startswith(cls.replace("/", "\\") + "\\")}
    files = list_images(folder)
    first = [f for f in files if f.name.startswith("train_")]
    needs_id = [f for f in files if "(needs_id)" in species.get(f.name, "") and "adult" not in species.get(f.name, "")]
    random.Random(42).shuffle(needs_id)
    dest = RESERVE / f"{cls}_swapped_out"
    dest.mkdir(parents=True, exist_ok=True)
    moved = (first + needs_id)[:n]
    for f in moved:
        shutil.move(str(f), str(dest / f.name))
        append_csv(MOVES_FILE, ["file", "class", "moved_to"],
                   {"file": f.name, "class": pest, "moved_to": str(dest.relative_to(DATASET_DIR.parent.parent))})
    log(f"{cls}: moved {len(moved)} photos out to {dest.name} "
        f"({sum(f in first for f in moved)} first-download, {sum(f not in first for f in moved)} needs-ID)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True, choices=sorted(SPECIES))
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--swap", type=int, default=0, help="first move N caterpillar/damage photos to reserve")
    a = ap.parse_args()
    crop, pest = a.cls.split("/")
    folder = RAW_DIR / a.cls
    if a.swap:
        swap_out(folder, a.cls, pest, a.swap)
    need = a.target - len(list_images(folder))
    if need <= 0:
        log(f"{a.cls}: already at target")
        return
    judge, done = Judge(), labeled()
    known = [phash(ImageOps.exif_transpose(Image.open(f)).convert("RGB")) for f in list_images(folder)]
    back = drawings = 0

    # 1. Moths kept in reserve (uncredited first-download moths are skipped).
    reserve = [f for f in list_images(RESERVE / f"{a.cls}_moths") if not f.name.startswith("train_")]
    random.Random(42).shuffle(reserve)
    for f in reserve:
        if back >= need:
            break
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        if is_drawing(judge, img):
            drawings += 1
            continue
        shutil.move(str(f), str(folder / f.name))
        append_csv(MOVES_FILE, ["file", "class", "moved_to"],
                   {"file": f.name, "class": pest, "moved_to": str(folder.relative_to(DATASET_DIR.parent.parent))})
        label_yes(str((folder / f.name).relative_to(RAW_DIR)), crop, pest, done)
        known.append(phash(img))
        back += 1
    log(f"{a.cls}: moved {back} moth photos back from reserve (skipped {drawings} drawings)")

    # 2. iNaturalist observations rejected as "moth" by inat_larvae.py.
    with open(REJECTED_FILE, newline="", encoding="utf-8") as fh:
        ids = [r["observation_id"] for r in csv.DictReader(fh) if r["class"] == a.cls and r["group"] == "moth"]
    have = {f.stem.replace("inat_", "") for f in list_images(folder)}
    ids = [i for i in ids if i not in have]
    added = dupes = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for i in range(0, len(ids), 100):
            if back + added >= need:
                break
            page = get_json(f"{API}?{urllib.parse.urlencode({'id': ','.join(ids[i:i + 100]), 'per_page': 100})}")["results"]
            time.sleep(1)
            todo = [(o, first_open_photo(o)) for o in page]
            todo = [(o, ph) for o, ph in todo if ph]
            results = pool.map(fetch_photo, [ph["url"].replace("/square.", "/medium.") for _, ph in todo])
            for (obs, photo), res in zip(todo, results):
                if res is None or back + added >= need:
                    continue
                img, h = res
                if is_duplicate(h, known) or is_drawing(judge, img):
                    dupes += 1
                    continue
                known.append(h)
                dest = folder / f"inat_{obs['id']}.jpg"
                img.save(dest, quality=92)
                rel = str(dest.relative_to(RAW_DIR))
                label_yes(rel, crop, pest, done)
                append_csv(CREDITS_FILE, CREDIT_FIELDS, {
                    "file": rel, "observation_id": obs["id"],
                    "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                    "species": f"{SPECIES[a.cls]} ({obs.get('quality_grade', '')}, adult)",
                    "license": photo["license_code"], "attribution": photo.get("attribution", ""),
                    "observed_on": obs.get("observed_on") or "", "place": obs.get("place_guess") or ""})
                added += 1
    total = len(list_images(folder))
    log(f"{a.cls}: downloaded {added} iNaturalist moth photos (skipped {dupes} duplicates/drawings) -> now {total}"
        + ("" if total >= a.target else "  (no more moth photos available)"))
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        raise
