"""Download research-grade, openly licensed pest photos from iNaturalist into ml/dataset/raw/.

Research grade means at least two iNaturalist identifiers agreed on the species, so
these labels are far more reliable than the CLIP-sorted photos. Classes with few
research-grade photos also take "needs ID" observations (NEEDS_ID), credited with
"(needs_id)" after the species name - check a sample by eye. Only one photo per
observation is taken (several photos of the same insect would be near-duplicates),
and only licences that allow reuse with credit: CC0, CC BY, CC BY-NC, CC BY-SA,
CC BY-NC-SA. No-derivatives (ND) and all-rights-reserved photos are skipped.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/inat_download.py --dry-run
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/inat_download.py

Each class is filled up to --target images (counting what's already in raw/).
Every downloaded photo is:
  - saved as ml/dataset/raw/<crop>/<class>/inat_<observation id>.jpg
  - credited in ml/dataset/inaturalist_credits.csv (cite these in your paper)
  - labeled bug_visible=yes in ml/dataset/labels.csv, since an iNaturalist
    observation is a photo of the insect itself
Re-running skips observations already downloaded.
"""
import argparse
import csv
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image, ImageOps
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "ml" / "dataset"
RAW_DIR = DATASET_DIR / "raw"
LABELS_FILE = DATASET_DIR / "labels.csv"
CREDITS_FILE = DATASET_DIR / "inaturalist_credits.csv"
LOG_FILE = Path(__file__).resolve().parent / "inat_download.log"

API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "pest-assessment-school-project/1.0 (open-licence photo downloader)"
LICENSES = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DUPLICATE_DISTANCE = 4
LARVA = {"term_id": 1, "term_value_id": 6}
NEEDS_ID = {"quality_grade": "needs_id"}  # species named but not yet confirmed; credited as such
LOOKALIKES = {"Laodelphax striatellus", "Sogatella furcifera", "Nilaparvata lugens"}  # stand-ins, credited as such

# class -> list of (species, extra query filters). Earlier entries are fetched first.
TARGETS = {
    "corn/fall_armyworm": [("Spodoptera frugiperda", LARVA),  # caterpillars first: what farmers see on corn
                           ("Spodoptera frugiperda", {})],
    "corn/common_cutworm": [("Spodoptera litura", LARVA),
                            ("Spodoptera litura", {})],
    "corn/armyworm": [("Mythimna separata", LARVA),            # oriental armyworm
                      ("Mythimna separata", {})],
    "corn/corn_aphid": [("Rhopalosiphum maidis", {})],
    "corn/corn_planthopper": [("Peregrinus maidis", {}),
                              ("Peregrinus maidis", NEEDS_ID),  # few research-grade photos exist
                              # Look-alikes from the same family (Delphacidae), added because every
                              # P. maidis photo is already used; credited as "(look-alike)".
                              ("Laodelphax striatellus", {}), ("Laodelphax striatellus", NEEDS_ID),
                              ("Sogatella furcifera", {}), ("Sogatella furcifera", NEEDS_ID),
                              ("Nilaparvata lugens", {}), ("Nilaparvata lugens", NEEDS_ID)],
}

CREDIT_FIELDS = ["file", "observation_id", "observation_url", "species", "license",
                 "attribution", "observed_on", "place"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=1000, help="fill each class up to this many images")
    p.add_argument("--dry-run", action="store_true", help="show how many would be downloaded, download nothing")
    p.add_argument("--class", dest="only", help="one class only, e.g. corn/fall_armyworm")
    p.add_argument("--workers", type=int, default=6, help="photos downloaded at the same time (default 6)")
    return p.parse_args()


def log(msg):
    tqdm.write(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)


def get_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def query_url(species, extra, id_above=0, per_page=200):
    params = {"taxon_name": species, "quality_grade": "research", "photos": "true",
              "photo_license": ",".join(sorted(LICENSES)), "order_by": "id", "order": "asc",
              "id_above": id_above, "per_page": per_page, **extra}
    return f"{API}?{urllib.parse.urlencode(params)}"


def available(species, extra):
    return get_json(query_url(species, extra, per_page=0))["total_results"]


def list_images(folder):
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS] if folder.is_dir() else []


def phash(img):
    return int(str(imagehash.phash(img)), 16)


def is_duplicate(h, known):
    return any(bin(h ^ k).count("1") <= DUPLICATE_DISTANCE for k in known)


def load_done_ids():
    if not CREDITS_FILE.exists():
        return set()
    with open(CREDITS_FILE, newline="", encoding="utf-8") as f:
        return {row["observation_id"] for row in csv.DictReader(f)}


def append_csv(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def first_open_photo(obs):
    for photo in obs.get("photos", []):
        if photo.get("license_code") in LICENSES and photo.get("url"):
            return photo
    return None


def fetch_photo(url):
    """Download and decode one photo; runs in a worker thread. Returns (img, hash) or None."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(get_bytes(url)))).convert("RGB")
        return img, phash(img)
    except Exception:
        return None


def fill_class(cls, sources, target, done_ids, pool, workers):
    crop, pest = cls.split("/")
    folder = RAW_DIR / cls
    folder.mkdir(parents=True, exist_ok=True)
    existing = list_images(folder)
    need = target - len(existing)
    if need <= 0:
        log(f"{cls}: already {len(existing)}, nothing to do")
        return 0

    known = []
    for f in existing:
        try:
            known.append(phash(ImageOps.exif_transpose(Image.open(f)).convert("RGB")))
        except Exception:
            pass

    on_inat = 0
    for species, extra in sources:
        if extra != LARVA:  # the larva query is a subset of the plain species query
            on_inat += available(species, extra)
            time.sleep(1)
    left_on_inat = max(0, on_inat - sum(1 for f in existing if f.name.startswith("inat_")))

    added, dupes, failed = 0, 0, 0
    bar = tqdm(total=min(need, left_on_inat), desc=f"{cls:<24}", unit="photo", dynamic_ncols=True)
    for species, extra in sources:
        id_above = 0
        while added < need:
            page = get_json(query_url(species, extra, id_above))["results"]
            time.sleep(1)  # iNaturalist asks for about 1 API request per second
            if not page:
                break
            id_above = page[-1]["id"]
            candidates = []
            for obs in page:
                photo = first_open_photo(obs)
                if str(obs["id"]) not in done_ids and photo is not None:
                    candidates.append((obs, photo))

            i = 0
            while i < len(candidates) and added < need:
                # Download only a few more than still needed, so the last class doesn't overshoot.
                chunk = candidates[i:i + (need - added) + workers]
                i += len(chunk)
                urls = [p["url"].replace("/square.", "/medium.") for _, p in chunk]
                for (obs, photo), result in zip(chunk, pool.map(fetch_photo, urls)):
                    if added >= need:
                        break
                    if result is None:
                        failed += 1
                        continue
                    img, h = result
                    oid = str(obs["id"])
                    done_ids.add(oid)
                    if is_duplicate(h, known):
                        dupes += 1
                        continue
                    known.append(h)

                    dest = folder / f"inat_{oid}.jpg"
                    img.save(dest, quality=92)
                    rel = str(dest.relative_to(RAW_DIR))
                    append_csv(LABELS_FILE, ["path", "crop", "pest_class", "bug_visible"],
                               {"path": rel, "crop": crop, "pest_class": pest, "bug_visible": "yes"})
                    append_csv(CREDITS_FILE, CREDIT_FIELDS, {
                        "file": rel, "observation_id": oid,
                        "observation_url": f"https://www.inaturalist.org/observations/{oid}",
                        "species": species + {(False, False): "", (True, False): " (needs_id)",
                                              (False, True): " (look-alike)", (True, True): " (look-alike, needs_id)"}[
                            (extra == NEEDS_ID, species in LOOKALIKES)],
                        "license": photo["license_code"],
                        "attribution": photo.get("attribution", ""),
                        "observed_on": obs.get("observed_on") or "", "place": obs.get("place_guess") or ""})
                    added += 1
                    bar.update(1)
                    bar.set_postfix(duplicates=dupes, failed=failed)
        if added >= need:
            break
    bar.close()

    note = "" if added >= need else "  (no more photos available on iNaturalist)"
    log(f"{cls}: added {added}, skipped {dupes} duplicates, {failed} failed -> now {len(existing) + added}{note}")
    return added


def main():
    args = parse_args()
    classes = [args.only] if args.only else list(TARGETS)

    if args.dry_run:
        print(f"{'class':<26}{'have':>6}{'need':>6}{'available':>11}")
        for cls in classes:
            have = len(list_images(RAW_DIR / cls))
            total = 0
            for species, extra in TARGETS[cls]:
                if extra != LARVA:  # the larva query is a subset of the species query
                    total += available(species, extra)
                    time.sleep(1)
            print(f"{cls:<26}{have:>6}{max(0, args.target - have):>6}{total:>11}")
        print("\nDry run, nothing downloaded.")
        return

    done_ids = load_done_ids()
    total = 0
    log(f"--- started {time.strftime('%Y-%m-%d %H:%M')}, {args.workers} downloads at a time")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for cls in classes:
            total += fill_class(cls, TARGETS[cls], args.target, done_ids, pool, args.workers)
    log(f"DONE: downloaded {total} photo(s). Credits: {CREDITS_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        raise
