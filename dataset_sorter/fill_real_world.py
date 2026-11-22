"""Add real-world "not corn" photos so the corn check learns more than "corn vs rice/banana".

Why: on 64 random phone photos (rooms, garden plants, the ground), the corn check said "corn" for
62 (22 Nov 2026). Its "not corn" side held only crop photos from a few research collections, so
everyday things and garden plants landed on the corn side.

Adds two folders (nothing existing is changed):
  ml/dataset/not_corn/ph_plants/   plants photographed in the Philippines, iNaturalist
                                   (research grade, open licences, corn = genus Zea left out).
                                   At most PER_SPECIES photos per species, newest observations
                                   first (mostly phone photos). Credited in inaturalist_credits.csv.
  ml/dataset/not_corn/everyday/    everyday scenes (rooms, streets, food, people, vehicles) from the
                                   COCO 2017 test images, Flickr photos with open licences ("no
                                   derivatives" licences left out). Credited in dataset_credits.csv.

Duplicates (perceptual hash) are skipped. Both credit files are backed up before the first write.
iNaturalist is asked about once per second, with the project's generic User-Agent.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/fill_real_world.py --dry-run
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/fill_real_world.py
    (--plants N --everyday N to change the 2,500 + 2,500 targets; safe to run again: it tops up)
"""
import argparse
import io
import json
import random
import shutil
import time
import urllib.parse
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from inat_download import (API, CREDIT_FIELDS as INAT_FIELDS, CREDITS_FILE as INAT_CREDITS, DATASET_DIR, LICENSES,
                           append_csv, fetch_photo, get_bytes, get_json, is_duplicate, list_images, phash)

NOT_CORN = DATASET_DIR / "not_corn"
PLANTS_DIR = NOT_CORN / "ph_plants"
EVERYDAY_DIR = NOT_CORN / "everyday"
CREDITS = DATASET_DIR / "dataset_credits.csv"
CREDIT_FIELDS = ["file", "source", "original_name", "licence", "citation"]
DOWNLOADS = Path(__file__).resolve().parent / "downloads" / "coco"
BACKUP_DIR = DATASET_DIR.parents[1] / "_archive" / "5_backups_and_logs"

PHILIPPINES = 6873  # iNaturalist place id
ZEA = 48449         # iNaturalist taxon id of the corn genus: never "not corn"
PER_SPECIES = 2
SEED = 42

COCO_INFO = "http://images.cocodataset.org/annotations/image_info_test2017.zip"
COCO_IMAGES = "http://images.cocodataset.org/test2017/"
COCO_CITATION = ("Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014. "
                 "https://cocodataset.org. Photo from Flickr: {flickr}")
# COCO licence ids -> name; 3 (BY-NC-ND) and 6 (BY-ND) forbid changes and are skipped.
COCO_LICENCES = {1: "CC BY-NC-SA 2.0", 2: "CC BY-NC 2.0", 4: "CC BY 2.0", 5: "CC BY-SA 2.0",
                 7: "No known copyright restrictions", 8: "United States Government Work"}


def backup(path):
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / f"{path.stem}_backup_before_real_world_{datetime.now():%Y%m%d_%H%M}.csv"
        if not dest.exists():
            shutil.copy2(path, dest)
            print(f"backed up {path.name} -> {dest.relative_to(DATASET_DIR.parents[1])}")


def known_hashes(folders):
    out = []
    for folder in folders:
        for f in list_images(folder):
            try:
                with Image.open(f) as im:
                    out.append(phash(ImageOps.exif_transpose(im).convert("RGB")))
            except Exception:
                pass
    return out


# ---------- Philippine plants (iNaturalist) ----------

PLANT_LIST = Path(__file__).resolve().parent / "downloads" / "inat_ph_plants.json"


def plant_candidates():
    """Every openly licensed research-grade plant observation in the Philippines (not Zea).
    The list takes ~15 minutes to fetch, so it is saved to downloads/ and reused."""
    if PLANT_LIST.exists():
        return [tuple(x) for x in json.loads(PLANT_LIST.read_text(encoding="utf-8"))]
    out = _fetch_plant_candidates()
    PLANT_LIST.parent.mkdir(parents=True, exist_ok=True)
    PLANT_LIST.write_text(json.dumps(out), encoding="utf-8")
    return out


def _fetch_plant_candidates():
    params = {"place_id": PHILIPPINES, "iconic_taxa": "Plantae", "without_taxon_id": ZEA,
              "quality_grade": "research", "photos": "true", "photo_license": ",".join(sorted(LICENSES)),
              "order_by": "id", "order": "desc", "per_page": 200}
    out, id_below = [], None
    while True:
        q = dict(params, **({"id_below": id_below} if id_below else {}))
        page = get_json(f"{API}?{urllib.parse.urlencode(q)}")["results"]
        time.sleep(1)  # iNaturalist asks for about 1 request per second
        if not page:
            return out
        for obs in page:
            photo = next((p for p in obs.get("photos", []) if p.get("license_code") in LICENSES and p.get("url")), None)
            taxon = obs.get("taxon") or {}
            if photo and taxon.get("id"):
                keep = {"id": obs["id"], "taxon": {"id": taxon["id"], "name": taxon.get("name", "")},
                        "observed_on": obs.get("observed_on"), "place_guess": obs.get("place_guess")}
                out.append((keep, {k: photo.get(k) for k in ("url", "license_code", "attribution")}))
        id_below = page[-1]["id"]
        print(f"  {len(out)} plant observations listed", end="\r", flush=True)


def pick_spread(cands, target, per_species):
    """Round-robin over species (newest first within each), at most per_species each."""
    by_species = defaultdict(list)
    for obs, photo in cands:
        by_species[obs["taxon"]["id"]].append((obs, photo))
    species = list(by_species)
    random.Random(SEED).shuffle(species)
    picks = []
    for round_ in range(per_species):
        for s in species:
            if len(by_species[s]) > round_:
                picks.append(by_species[s][round_])
    return picks if target is None else picks[: target * 2]  # spare ones for duplicates / failures


def fill_plants(target, dry_run, workers):
    PLANTS_DIR.mkdir(parents=True, exist_ok=True)
    have = len(list_images(PLANTS_DIR))
    need = target - have
    print(f"\nPhilippine plants: {have} already, {max(need, 0)} to add")
    if need <= 0:
        return
    cands = plant_candidates()
    species = len({o["taxon"]["id"] for o, _ in cands})
    print(f"\n  {len(cands)} observations of {species} species available")
    picks = [(o, p) for o, p in pick_spread(cands, None, PER_SPECIES)
             if not (PLANTS_DIR / f"inat_{o['id']}.jpg").exists()]
    if dry_run:
        print(f"  would download {need} of {len(picks)} picks")
        return
    backup(INAT_CREDITS)
    known = known_hashes([PLANTS_DIR])
    added = 0
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(picks), workers * 4):
            if added >= need:
                break
            chunk = picks[i: i + workers * 4]
            got = pool.map(fetch_photo, [p["url"].replace("/square.", "/medium.") for _, p in chunk])
            for (obs, photo), res in zip(chunk, got):
                if res is None or added >= need:
                    continue
                img, h = res
                if is_duplicate(h, known):
                    continue
                known.append(h)
                name = f"inat_{obs['id']}.jpg"
                img.save(PLANTS_DIR / name, quality=92)
                append_csv(INAT_CREDITS, INAT_FIELDS, {
                    "file": f"not_corn\\ph_plants\\{name}", "observation_id": obs["id"],
                    "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                    "species": obs["taxon"].get("name", ""), "license": photo["license_code"],
                    "attribution": photo.get("attribution", ""), "observed_on": obs.get("observed_on") or "",
                    "place": obs.get("place_guess") or ""})
                added += 1
            print(f"  added {added}/{need}", end="\r", flush=True)
            time.sleep(1)
    print(f"\n  Philippine plants: added {added}")


# ---------- Everyday scenes (COCO) ----------

def coco_list():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    zpath = DOWNLOADS / "image_info_test2017.zip"
    if not zpath.exists():
        zpath.write_bytes(get_bytes(COCO_INFO))
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith("image_info_test2017.json"))
        info = json.loads(z.read(name))
    return [im for im in info["images"] if im.get("license") in COCO_LICENCES]


def fetch_coco(im):
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(get_bytes(COCO_IMAGES + im["file_name"])))).convert("RGB")
        return img, phash(img)
    except Exception:
        return None


def fill_everyday(target, dry_run, workers):
    EVERYDAY_DIR.mkdir(parents=True, exist_ok=True)
    have = len(list_images(EVERYDAY_DIR))
    need = target - have
    print(f"\nEveryday scenes: {have} already, {max(need, 0)} to add")
    if need <= 0:
        return
    images = coco_list()
    random.Random(SEED).shuffle(images)
    images = [im for im in images if not (EVERYDAY_DIR / f"coco_{im['file_name']}").exists()]
    print(f"  {len(images)} openly licensed COCO photos to choose from")
    if dry_run:
        print(f"  would download {need}")
        return
    backup(CREDITS)
    known = known_hashes([EVERYDAY_DIR])
    added = 0
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(images), workers * 4):
            if added >= need:
                break
            chunk = images[i: i + workers * 4]
            for im, res in zip(chunk, pool.map(fetch_coco, chunk)):
                if res is None or added >= need:
                    continue
                img, h = res
                if is_duplicate(h, known):
                    continue
                known.append(h)
                name = f"coco_{im['file_name']}"
                img.save(EVERYDAY_DIR / name, quality=92)
                append_csv(CREDITS, CREDIT_FIELDS, {
                    "file": f"not_corn\\everyday\\{name}", "source": "COCO 2017 test images",
                    "original_name": im["file_name"], "licence": COCO_LICENCES[im["license"]],
                    "citation": COCO_CITATION.format(flickr=im.get("flickr_url", ""))})
                added += 1
            print(f"  added {added}/{need}", end="\r", flush=True)
    print(f"\n  Everyday scenes: added {added}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plants", type=int, default=2500)
    ap.add_argument("--everyday", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fill_everyday(a.everyday, a.dry_run, a.workers)
    fill_plants(a.plants, a.dry_run, a.workers)


if __name__ == "__main__":
    main()
