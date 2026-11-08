"""Fill corn classes in ml/dataset/raw/ up to --target images from free public datasets.

Sources (all downloadable without an account):
  IP102, Hugging Face copy            Asian corn borer
  Mendeley Data (CCMT, Ghana)         corn healthy and disease photos (its API gives max 100 per folder)
  PlantVillage, GitHub                corn healthy and disease leaves

The IP102 and PlantVillage files must already be in dataset_sorter/downloads/.
Mendeley photos are fetched while importing.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/bulk_import.py --dry-run
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/bulk_import.py

healthy and unidentified_damage photos are labeled bug_visible=no (healthy leaves and
plant diseases, no insects). IP102 pest photos are NOT labeled, because IP102 mixes
insect photos with damage-only photos (deadhearts, hopperburn, bored stalks) - label
those with ml/src/label_bug_visible.py.

Every imported photo is credited in ml/dataset/dataset_credits.csv. Re-running only
adds what's still missing.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm import tqdm

SORTER_DIR = Path(__file__).resolve().parent
DOWNLOADS = SORTER_DIR / "downloads"
DATASET_DIR = SORTER_DIR.parent / "ml" / "dataset"
RAW_DIR = DATASET_DIR / "raw"
LABELS_FILE = DATASET_DIR / "labels.csv"
CREDITS_FILE = DATASET_DIR / "dataset_credits.csv"
LOG_FILE = SORTER_DIR / "bulk_import.log"

UA = {"User-Agent": "Mozilla/5.0 (pest-assessment-school-project)"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DUPLICATE_DISTANCE = 4
MAX_SIDE = 1024  # imported photos are shrunk to this; training resizes to 256 anyway
CREDIT_FIELDS = ["file", "source", "original_name", "licence", "citation"]


# ---------------------------------------------------------------- helpers

def log(msg):
    tqdm.write(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def urlopen(url, headers=None, timeout=120):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)


def get_json(url):
    with urlopen(url) as r:
        return json.load(r)


def phash(img):
    return int(str(imagehash.phash(img)), 16)


def is_duplicate(h, known):
    return any(bin(h ^ k).count("1") <= DUPLICATE_DISTANCE for k in known)


def list_images(folder):
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS] if folder.is_dir() else []


def append_csv(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- sources
# Each source yields (original_name, image_bytes) in a shuffled order.

class ParquetSource:
    """Rows of downloaded Hugging Face parquet files whose label is in `labels` (names or ids)."""

    def __init__(self, key, folder, pattern, labels, licence, citation):
        self.key, self.licence, self.citation = key, licence, citation
        self.files = sorted((DOWNLOADS / folder).glob(pattern))
        self.labels = labels
        self.desc = f"{folder}/{pattern} {labels}"

    def _label_ids(self, pf):
        meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]
        names = meta.get("names")
        return {names.index(l) if isinstance(l, str) else l for l in self.labels}

    def candidates(self, rng, want):
        groups = []
        for f in self.files:
            pf = pq.ParquetFile(f)
            ids = self._label_ids(pf)
            for rg in range(pf.num_row_groups):
                labels = pf.read_row_group(rg, columns=["label"]).column(0).to_pylist()
                rows = [i for i, l in enumerate(labels) if l in ids]
                if rows:
                    groups.append((f, rg, rows))
        rng.shuffle(groups)
        for f, rg, rows in groups:
            images = pq.ParquetFile(f).read_row_group(rg, columns=["image"]).column(0)
            rng.shuffle(rows)
            for i in rows:
                im = images[i].as_py()
                yield (im.get("path") or f"{f.stem}_{rg}_{i}"), im["bytes"]


class MendeleySource:
    """One folder of a Mendeley Data dataset (the public API lists at most 100 files per folder)."""

    def __init__(self, key, dataset, version, folder_path, licence, citation):
        self.key, self.licence, self.citation = key, licence, citation
        self.dataset, self.version, self.folder_path = dataset, version, folder_path
        self.desc = f"Mendeley {dataset}: {folder_path}"

    def _folder_id(self):
        folders = get_json(f"https://data.mendeley.com/public-api/datasets/{self.dataset}/folders/{self.version}")
        by_id = {f["id"]: f for f in folders}

        def path(f):
            parts = [f["name"]]
            while f.get("parent_id") in by_id:
                f = by_id[f["parent_id"]]
                parts.append(f["name"])
            return "/".join(reversed(parts))

        for f in folders:
            if path(f) == self.folder_path:
                return f["id"]
        raise ValueError(f"folder not found in {self.dataset}: {self.folder_path}")

    def candidates(self, rng, want):
        fid = self._folder_id()
        files = get_json(f"https://data.mendeley.com/api/datasets/{self.dataset}/files"
                         f"?version={self.version}&folder_id={fid}")
        files = [f for f in files if Path(f["filename"]).suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(files)

        def fetch(f):
            try:
                with urlopen(f["content_details"]["download_url"]) as r:
                    return f["filename"], r.read()
            except Exception:
                return f["filename"], None

        with ThreadPoolExecutor(6) as pool:
            for i in range(0, len(files), 12):
                for name, data in pool.map(fetch, files[i:i + 12]):
                    if data:
                        yield name, data


class DirSource:
    """Image files in a local folder."""

    def __init__(self, key, folder, licence, citation):
        self.key, self.licence, self.citation = key, licence, citation
        self.folder = DOWNLOADS / folder
        self.desc = str(folder)

    def candidates(self, rng, want):
        files = list_images(self.folder)
        rng.shuffle(files)
        for f in files:
            yield f.name, f.read_bytes()


# ---------------------------------------------------------------- what goes where

IP102 = dict(licence="free for academic use (IP102); Hugging Face copy MIT",
             citation="Wu et al., IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition, CVPR 2019. "
                      "Copy: https://huggingface.co/datasets/EnmmmmOvO/insect-pest-dataset")
CCMT = dict(licence="CC BY 4.0",
            citation="Mensah et al., CCMT: Dataset for crop pest and disease detection, Data in Brief 2023, doi:10.17632/bwh3zbpkpv.1")
PLANTVILLAGE = dict(licence="CC BY-SA 3.0",
                    citation="Hughes & Salathe, An open access repository of images on plant health (PlantVillage), "
                             "arXiv:1511.08060, https://github.com/spMohanty/PlantVillage-Dataset")
AGML_CORN = dict(licence="CC BY 4.0",
                 citation="Tchokogoue et al. (2024), A Dataset for Early Detection of Corn Leaf Pests in Precision "
                          "Agriculture, Mendeley Data, doi:10.17632/ymvghfcww7.1 (via Project-AgML)")
PV = "plantvillage/raw/color/Corn_(maize)___"
CCMT_MAIZE = "Raw Data/CCMT Dataset/Maize/"


def plan():
    """class -> list of (source, quota). quota None = fill whatever is still missing."""
    ip102 = lambda labels: ParquetSource("ip102", "ip102_hf", "*.parquet", labels, **IP102)
    return {
        # Field photos of fall armyworm on corn leaves (Cameroon); run after inat_download fills
        # the class to 3,500 so these field photos make up the last 1,500.
        "corn/fall_armyworm": [(ParquetSource("agmlCorn", "agml_corn_pest", "raw_*.parquet",
                                              ["spodoptera_frugiperda"], **AGML_CORN), None)],
        "corn/corn_aphid": [(ip102([24]), None)],             # IP102 "aphids" (corn section)
        "corn/armyworm": [(ip102([23]), None)],               # IP102 "army worm" (corn section)
        "corn/healthy": [
            (MendeleySource("ccmt", "bwh3zbpkpv", 1, CCMT_MAIZE + "healthy", **CCMT), 100),
            (DirSource("plantvillage", PV + "healthy", **PLANTVILLAGE), None),
        ],
        "corn/unidentified_damage": [
            *[(MendeleySource("ccmt", "bwh3zbpkpv", 1, CCMT_MAIZE + d, **CCMT), 100)
              for d in ["leaf blight", "leaf spot", "streak virus"]],
            (DirSource("plantvillage", PV + "Common_rust_", **PLANTVILLAGE), 250),
            (DirSource("plantvillage", PV + "Northern_Leaf_Blight", **PLANTVILLAGE), 250),
            (DirSource("plantvillage", PV + "Cercospora_leaf_spot Gray_leaf_spot", **PLANTVILLAGE), None),
        ],
    }


# ---------------------------------------------------------------- import

def prepare(data):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def fill_class(cls, sources, target, rng):
    crop, pest = cls.split("/")
    folder = RAW_DIR / cls
    folder.mkdir(parents=True, exist_ok=True)
    existing = list_images(folder)
    need = target - len(existing)
    if need <= 0:
        log(f"{cls}: already {len(existing)}, nothing to do")
        return 0
    label_no = pest in ("healthy", "unidentified_damage")

    # Photos already moved to ml/dataset/rejected/ count as duplicates, so a refill can't
    # bring a rejected photo (e.g. a diagram) back in.
    known = []
    for f in existing + list_images(DATASET_DIR / "rejected" / cls):
        try:
            known.append(phash(prepare(f.read_bytes())))
        except Exception:
            pass

    added, dupes, failed = 0, 0, 0
    bar = tqdm(total=need, desc=f"{cls:<26}", unit="photo", dynamic_ncols=True)
    for source, quota in sources:
        take = min(quota or need, need - added)
        if take <= 0:
            break
        got = 0
        try:
            for name, data in source.candidates(rng, take):
                if got >= take:
                    break
                try:
                    img = prepare(data)
                except Exception:
                    failed += 1
                    continue
                h = phash(img)
                if is_duplicate(h, known):
                    dupes += 1
                    continue
                known.append(h)
                stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(name).stem)[:40]
                short = hashlib.sha1(data).hexdigest()[:6]
                dest = folder / f"{source.key}_{stem}_{short}.jpg"
                img.save(dest, quality=92)
                rel = str(dest.relative_to(RAW_DIR))
                if label_no:
                    append_csv(LABELS_FILE, ["path", "crop", "pest_class", "bug_visible"],
                               {"path": rel, "crop": crop, "pest_class": pest, "bug_visible": "no"})
                append_csv(CREDITS_FILE, CREDIT_FIELDS, {"file": rel, "source": source.desc, "original_name": name,
                                                         "licence": source.licence, "citation": source.citation})
                got += 1
                added += 1
                bar.update(1)
                bar.set_postfix(duplicates=dupes, failed=failed)
        except Exception as e:
            log(f"  {cls}: source failed ({source.desc}): {e}")
        if got < take:
            log(f"  {cls}: {source.desc} gave {got} of {take}")
    bar.close()
    note = "" if added >= need else f"  (sources ran out, still {need - added} short)"
    log(f"{cls}: added {added}, skipped {dupes} duplicates, {failed} unreadable -> now {len(existing) + added}{note}")
    return added


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=1000, help="fill each class up to this many images")
    p.add_argument("--dry-run", action="store_true", help="show the plan, import nothing")
    p.add_argument("--class", dest="only", help="one class only, e.g. corn/healthy")
    return p.parse_args()


def main():
    args = parse_args()
    the_plan = plan()
    classes = [args.only] if args.only else list(the_plan)
    if args.dry_run:
        for cls in classes:
            have = len(list_images(RAW_DIR / cls))
            print(f"{cls}: have {have}, need {max(0, args.target - have)}")
            for source, quota in the_plan[cls]:
                print(f"    {'rest' if quota is None else quota:>5}  {source.desc}")
        print("\nDry run, nothing imported.")
        return
    rng = random.Random(42)
    log(f"--- started {time.strftime('%Y-%m-%d %H:%M')}")
    total = sum(fill_class(cls, the_plan[cls], args.target, rng) for cls in classes)
    log(f"DONE: imported {total} photo(s). Credits: {CREDITS_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        raise
