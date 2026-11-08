"""Fetch extra pest photos into a staging folder, check them, then import the good ones.

Sources (no account needed):
  commons   Wikimedia Commons species categories (CC0 / public domain / CC BY / CC BY-SA only)
  inat      iNaturalist photos not used yet: "needs ID" observations (not yet confirmed by a
            second person) and extra species of a genus. Must be checked by eye.

Photos are NOT put into ml/dataset/raw/ straight away. Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/stage_extra.py stage
        downloads into dataset_sorter/staging/<source>/<crop>/<class>/ and draws contact
        sheets in dataset_sorter/audit/staging_*.jpg
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/stage_extra.py import <source> <crop/class> [--unlabeled]
        moves that staged folder into raw/ (skipping duplicates and any file listed in
        dataset_sorter/staging/rejected.txt), labels it bug_visible=yes (or leaves it for
        label_bug_visible.py with --unlabeled), and credits it in ml/dataset/dataset_credits.csv
"""
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageOps

from inat_download import CREDITS_FILE as INAT_CREDITS, LICENSES as INAT_LICENSES, first_open_photo

SORTER_DIR = Path(__file__).resolve().parent
STAGING = SORTER_DIR / "staging"
STAGING_CREDITS = STAGING / "staging_credits.csv"
REJECTED = STAGING / "rejected.txt"
AUDIT = SORTER_DIR / "audit"
DATASET_DIR = SORTER_DIR.parent / "ml" / "dataset"
RAW_DIR = DATASET_DIR / "raw"
LABELS_FILE = DATASET_DIR / "labels.csv"
CREDITS_FILE = DATASET_DIR / "dataset_credits.csv"
UA = {"User-Agent": "pest-assessment-school-project/1.0 (open-licence photo downloader)"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
OPEN_LICENCES = ("cc0", "public domain", "cc by", "cc-by", "pd")  # NC and ND are excluded below
FIELDS = ["file", "source", "original_name", "licence", "citation"]

COMMONS = {  # Commons category -> class
    "Spodoptera_frugiperda": "corn/fall_armyworm",
    "Ostrinia_furnacalis": "corn/asian_corn_borer",
}
INAT = [  # (taxon, quality grade, class): "needs ID" photos, not yet confirmed by a second person
    ("Spodoptera frugiperda", "needs_id", "corn/fall_armyworm"),
    ("Ostrinia furnacalis", "needs_id", "corn/asian_corn_borer"),
]


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def get_bytes(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:  # 429 = too many requests: wait and retry
                raise
            time.sleep(30 * (attempt + 1))


def append_csv(path, row):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def is_staged(source, cls, name):
    return (STAGING / source / cls / f"{name}.jpg").exists()


def save_staged(source, cls, name, data, licence, citation):
    folder = STAGING / source / cls
    folder.mkdir(parents=True, exist_ok=True)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((1024, 1024))
    dest = folder / f"{name}.jpg"
    img.save(dest, quality=92)
    append_csv(STAGING_CREDITS, {"file": str(dest.relative_to(STAGING)), "source": source, "original_name": name,
                                 "licence": licence, "citation": citation})


def stage_commons():
    for cat, cls in COMMONS.items():
        # 960 px is one of Commons' standard thumbnail sizes; other sizes get rate-limited.
        # The API only returns thumbnail links for 50 files per request, so page through.
        q = {"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{cat}", "gcmtype": "file",
             "gcmlimit": 50, "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": 960, "format": "json"}
        pages, cont = [], {}
        while True:
            r = get_json("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({**q, **cont}))
            pages += list(r.get("query", {}).get("pages", {}).values())
            if "continue" not in r:
                break
            cont = r["continue"]
        n = 0
        for p in pages:
            if "imageinfo" not in p or is_staged("commons", cls, f"commons_{p['pageid']}"):
                continue
            info = p["imageinfo"][0]
            meta = info.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            low = lic.lower()
            if not low.startswith(OPEN_LICENCES) or "nc" in low.split("-") or "nd" in low.split("-") or \
                    info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            artist = meta.get("Artist", {}).get("value", "")
            artist = "".join(ch for ch in artist if ch != "\n")[:200]
            try:
                save_staged("commons", cls, f"commons_{p['pageid']}", get_bytes(info.get("thumburl") or info["url"]), lic,
                            f"Wikimedia Commons, {p['title']}, {artist}, {info['descriptionurl']}")
                n += 1
            except Exception as e:
                print("   failed", p["title"], e)
            time.sleep(1.5)  # be gentle: Commons rate-limits fast downloaders
        print(f"commons {cat} -> {cls}: staged {n}")


def stage_inat():
    done = set()
    if INAT_CREDITS.exists():
        with open(INAT_CREDITS, newline="", encoding="utf-8") as f:
            done = {r["observation_id"] for r in csv.DictReader(f)}
    for taxon, grade, cls in INAT:
        params = {"taxon_name": taxon, "quality_grade": grade, "photos": "true",
                  "photo_license": ",".join(sorted(INAT_LICENSES)), "order_by": "id", "order": "asc", "per_page": 200}
        id_above, n = 0, 0
        while True:
            page = get_json("https://api.inaturalist.org/v1/observations?" +
                            urllib.parse.urlencode({**params, "id_above": id_above}))["results"]
            time.sleep(1)
            if not page:
                break
            id_above = page[-1]["id"]
            for obs in page:
                oid = str(obs["id"])
                photo = first_open_photo(obs)
                if oid in done or photo is None or is_staged(f"inat_{grade}", cls, f"inat_{oid}"):
                    continue
                try:
                    data = get_bytes(photo["url"].replace("/square.", "/medium."))
                except Exception:
                    continue
                done.add(oid)
                species = (obs.get("taxon") or {}).get("name", taxon)
                save_staged(f"inat_{grade}", cls, f"inat_{oid}", data, photo["license_code"],
                            f"iNaturalist observation {oid} ({species}, {grade}), {photo.get('attribution', '')}, "
                            f"https://www.inaturalist.org/observations/{oid}")
                n += 1
                time.sleep(0.2)
        print(f"inat {taxon} ({grade}) -> {cls}: staged {n}")


def contact_sheets():
    AUDIT.mkdir(exist_ok=True)
    for folder in sorted(p for p in STAGING.glob("*/*/*") if p.is_dir()):
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            continue
        tile, per_row = 120, 12
        rows = -(-len(files) // per_row)
        sheet = Image.new("RGB", (per_row * tile, 24 + rows * tile), "white")
        d = ImageDraw.Draw(sheet)
        d.text((6, 6), f"{folder.relative_to(STAGING)}: {len(files)} photos (all shown)", fill="black")
        for k, f in enumerate(files):
            x, y = (k % per_row) * tile, 24 + (k // per_row) * tile
            sheet.paste(ImageOps.fit(Image.open(f).convert("RGB"), (tile - 4, tile - 4)), (x + 2, y + 2))
            d.rectangle((x + 2, y + 2, x + 30, y + 14), fill="black")
            d.text((x + 4, y + 3), str(k), fill="white")
        out = AUDIT / f"staging_{'_'.join(folder.relative_to(STAGING).parts)}.jpg"
        sheet.save(out, quality=85)
        print(f"sheet: {out.name} ({len(files)} photos)")


def import_staged(source, cls, label="yes"):
    """label="yes" for photos of the insect itself; label=None leaves them for
    ml/src/label_bug_visible.py (folders that mix in damage-only photos)."""
    folder = STAGING / source / cls
    rejected = set(REJECTED.read_text(encoding="utf-8").split()) if REJECTED.exists() else set()
    with open(STAGING_CREDITS, newline="", encoding="utf-8") as f:
        credits = {r["file"]: r for r in csv.DictReader(f)}
    dest_dir = RAW_DIR / cls
    known = []
    for f in dest_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                known.append(int(str(imagehash.phash(Image.open(f).convert("RGB"))), 16))
            except Exception:
                pass
    crop, pest = cls.split("/")
    added = dupes = skipped = 0
    for f in sorted(folder.iterdir()):
        rel_staged = str(f.relative_to(STAGING))
        if f.name in rejected or rel_staged in rejected:
            skipped += 1
            continue
        h = int(str(imagehash.phash(Image.open(f).convert("RGB"))), 16)
        if any(bin(h ^ k).count("1") <= 4 for k in known):
            dupes += 1
            continue
        known.append(h)
        dest = dest_dir / f.name
        f.replace(dest)
        rel = str(dest.relative_to(RAW_DIR))
        if label:
            with open(LABELS_FILE, "a", newline="", encoding="utf-8") as lf:
                csv.DictWriter(lf, fieldnames=["path", "crop", "pest_class", "bug_visible"]).writerow(
                    {"path": rel, "crop": crop, "pest_class": pest, "bug_visible": label})
        c = credits.get(rel_staged, {})
        append_csv(CREDITS_FILE, {"file": rel, "source": c.get("source", source), "original_name": c.get("original_name", f.stem),
                                  "licence": c.get("licence", ""), "citation": c.get("citation", "")})
        added += 1
    print(f"{source} -> {cls}: imported {added}, skipped {dupes} duplicates, {skipped} rejected")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "stage":
        stage_commons()
        stage_inat()
        contact_sheets()
    elif len(sys.argv) >= 2 and sys.argv[1] == "sheets":
        contact_sheets()
    elif len(sys.argv) in (4, 5) and sys.argv[1] == "import":
        import_staged(sys.argv[2], sys.argv[3], None if sys.argv[4:] == ["--unlabeled"] else "yes")
    else:
        sys.exit(__doc__)
