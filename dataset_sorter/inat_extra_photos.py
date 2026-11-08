"""Download every openly licensed photo of every iNaturalist observation of one species.

inat_download.py takes only the first photo of each observation. For rare species this adds
the other photos (same insect, other angles) as inat_<observation id>_p<N>.jpg, plus
"casual" observations (no date/place, or captive). ml/src/train.py keeps all photos of one
observation in the same split and fold, so the same insect is never in both train and test.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/inat_extra_photos.py --class corn/corn_planthopper --species "Peregrinus maidis"

Photos are credited in ml/dataset/inaturalist_credits.csv and left unlabeled (an extra photo
may show only the plant); run prelabel_bug_visible.py / label_bug_visible.py afterwards.
"""
import argparse
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

from inat_download import (API, CREDIT_FIELDS, CREDITS_FILE, LICENSES, RAW_DIR, append_csv, fetch_photo, get_json,
                           is_duplicate, list_images, log, phash)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    folder = RAW_DIR / a.cls
    have = {f.stem for f in list_images(folder)}
    known = [phash(ImageOps.exif_transpose(Image.open(f)).convert("RGB")) for f in list_images(folder)]
    todo = []  # (obs, photo, file stem)
    id_above = 0
    while True:
        q = {"taxon_name": a.species, "photos": "true", "order_by": "id", "order": "asc", "id_above": id_above,
             "per_page": 200}  # every quality grade
        page = get_json(f"{API}?{urllib.parse.urlencode(q)}")["results"]
        time.sleep(1)
        if not page:
            break
        id_above = page[-1]["id"]
        for obs in page:
            photos = [p for p in obs.get("photos", []) if p.get("license_code") in LICENSES and p.get("url")]
            for n, photo in enumerate(photos, 1):
                stem = f"inat_{obs['id']}" if n == 1 else f"inat_{obs['id']}_p{n}"
                if stem not in have:
                    todo.append((obs, photo, stem))
    log(f"--- {a.cls}: {len(todo)} more open-licence photos of {a.species} on iNaturalist")
    added = dupes = failed = 0
    with ThreadPoolExecutor(a.workers) as pool:
        results = pool.map(fetch_photo, [p["url"].replace("/square.", "/medium.") for _, p, _ in todo])
        for (obs, photo, stem), res in zip(todo, results):
            if res is None:
                failed += 1
                continue
            img, h = res
            if is_duplicate(h, known):
                dupes += 1
                continue
            known.append(h)
            dest = folder / f"{stem}.jpg"
            img.save(dest, quality=92)
            append_csv(CREDITS_FILE, CREDIT_FIELDS, {
                "file": str(dest.relative_to(RAW_DIR)), "observation_id": obs["id"],
                "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                "species": f"{a.species} ({obs.get('quality_grade', '')})", "license": photo["license_code"],
                "attribution": photo.get("attribution", ""),
                "observed_on": obs.get("observed_on") or "", "place": obs.get("place_guess") or ""})
            added += 1
    log(f"{a.cls}: added {added} photos, skipped {dupes} duplicates, {failed} failed -> now {len(list_images(folder))}")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        raise
