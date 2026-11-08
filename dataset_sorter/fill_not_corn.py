"""Add rice and banana plant photos to ml/dataset/not_corn/ so the corn check is balanced.

Sources, in order:
  1. The rice/banana classes archived in ml/dataset/archive_rice_banana/ (copied, not moved).
     Only credited photos. From pest folders, only photos labeled bug_visible=no (damage with
     no insect in view): the corn check compares plant photos, and insect close-ups in
     "not corn" would teach it that a pest close-up means "not corn".
  2. The archived Project-AgML rice parquet files in dataset_sorter/archive_non_corn/downloads/
     (the "Insect" class is left out for the same reason).
  3. CCMT raw field photos of cashew, cassava and tomato (Ghana), from the Kaggle copy
     nirmalsankalana/crop-pest-and-disease-detection (needs ~/.kaggle/kaggle.json). The zip
     is downloaded once to dataset_sorter/downloads/kaggle_ccmt/; maize photos are skipped.
     Picks are spread evenly over the three crops.
Duplicates are skipped, including flipped/turned copies; archived photos that are tilted
augmentation copies are skipped too. Every photo is credited in ml/dataset/dataset_credits.csv.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/fill_not_corn.py --target 10000 --dry-run
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/fill_not_corn.py --target 10000
"""
import argparse
import csv
import hashlib
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from api_import import Known, is_tilted_copy, variant_hashes
from bulk_import import (CREDIT_FIELDS, CREDITS_FILE, DATASET_DIR, ParquetSource, append_csv, list_images, log,
                         phash, prepare)

NOT_CORN = DATASET_DIR / "not_corn"
ARCHIVE = DATASET_DIR / "archive_rice_banana"
AGML = Path(__file__).resolve().parent / "archive_non_corn" / "downloads"
PEST_FOLDERS = {"rice": ["black_bug", "brown_planthopper", "stem_borer"], "banana": ["aphid", "thrips", "weevil"]}
AGML_SETS = [  # key, folder, pattern, {label: not_corn subfolder}, citation
    ("agmlRiceBGD", "agml_rice_bangladesh", "raw_*.parquet",
     {"Healthy": "rice_healthy", "Leaf Scald": "rice_unidentified_damage", "Rice Blast": "rice_unidentified_damage",
      "Rice Leaffolder": "rice_unidentified_damage", "Rice Stripes": "rice_unidentified_damage",
      "Rice Tungro": "rice_unidentified_damage"},
     "Project-AgML/rice_disease_classification_bangladesh, "
     "https://huggingface.co/datasets/Project-AgML/rice_disease_classification_bangladesh"),
    ("agmlRiceBD", "agml_rice_bd", "*.parquet",
     {"healthy": "rice_healthy", "bacterial_leaf_blight": "rice_unidentified_damage",
      "brown_spot": "rice_unidentified_damage", "leaf_blast": "rice_unidentified_damage",
      "leaf_scald": "rice_unidentified_damage", "narrow_brown_spot": "rice_unidentified_damage"},
     "Project-AgML/rice_leaf_disease_classification_bd, "
     "https://huggingface.co/datasets/Project-AgML/rice_leaf_disease_classification_bd"),
]


CCMT_KAGGLE = "nirmalsankalana/crop-pest-and-disease-detection"
CCMT_ZIP = Path(__file__).resolve().parent / "downloads" / "kaggle_ccmt" / "crop-pest-and-disease-detection.zip"
CCMT_CROPS = ("cashew", "cassava", "tomato")
CCMT_CITE = ("Mensah et al., CCMT: Dataset for crop pest and disease detection, Data in Brief 2023, "
             "doi:10.17632/bwh3zbpkpv.1 (raw images, via Kaggle copy " + CCMT_KAGGLE + ")")


def ccmt_zip():
    """Download the CCMT Kaggle copy once (streamed to disk)."""
    if not CCMT_ZIP.exists():
        from tqdm import tqdm
        from api_import import kaggle_auth, request
        CCMT_ZIP.parent.mkdir(parents=True, exist_ok=True)
        tmp = CCMT_ZIP.with_suffix(".part")
        with request(f"https://www.kaggle.com/api/v1/datasets/download/{CCMT_KAGGLE}", kaggle_auth(), timeout=600) as r, \
                open(tmp, "wb") as f, tqdm(total=int(r.headers.get("Content-Length") or 0), unit="B",
                                           unit_scale=True, desc="download CCMT") as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(CCMT_ZIP)
    return CCMT_ZIP


def ccmt_candidates(rng):
    """Yield (crop, member name, not_corn subfolder) round-robin over cashew/cassava/tomato."""
    import zipfile
    z = zipfile.ZipFile(ccmt_zip())
    per_crop = {c: [] for c in CCMT_CROPS}
    for n in z.namelist():
        if not n.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        cls = n.split("/")[-2] if "/" in n else ""
        crop = cls.split()[0].lower() if cls else ""
        if crop in per_crop:
            sub = f"{crop}_healthy" if "healthy" in cls.lower() else f"{crop}_unidentified_damage"
            per_crop[crop].append((n, sub, cls))
    for c in CCMT_CROPS:
        rng.shuffle(per_crop[c])
    log(f"  CCMT zip: {', '.join(f'{c} {len(v)}' for c, v in per_crop.items())} photos")
    for i in range(max(len(v) for v in per_crop.values())):
        for c in CCMT_CROPS:
            if i < len(per_crop[c]):
                n, sub, cls = per_crop[c][i]
                yield z, n, sub, cls


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def archive_candidates():
    """(file, not_corn subfolder, credit row) for credited archived plant photos."""
    norm = lambda p: p.replace("/", "\\")
    credits = {norm(r["file"]): r for r in read_csv(ARCHIVE / "logs" / "dataset_credits.csv")}
    for r in read_csv(ARCHIVE / "logs" / "inaturalist_credits.csv"):
        credits[norm(r["file"])] = {"source": f"iNaturalist observation {r['observation_id']}",
                                    "original_name": r["observation_url"], "licence": r["license"],
                                    "citation": f"{r['attribution']}, iNaturalist ({r['species']})"}
    labels = {norm(r["path"]): r["bug_visible"] for r in read_csv(ARCHIVE / "logs" / "labels.csv")}
    out = []
    for crop in ("rice", "banana"):
        folders = [("healthy", f"{crop}_healthy"), ("unidentified_damage", f"{crop}_unidentified_damage")]
        folders += [(p, f"{crop}_unidentified_damage") for p in PEST_FOLDERS[crop]]
        for cls, dest in folders:
            for part in ("raw", "reserve"):
                for f in list_images(ARCHIVE / part / crop / cls):
                    rel = f"{crop}\\{cls}\\{f.name}"
                    is_pest = cls in PEST_FOLDERS[crop]
                    if rel in credits and (not is_pest or labels.get(rel) == "no"):
                        out.append((f, dest, credits[rel]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=10000, help="total not_corn photos to aim for")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--steps", default="archive,agml,ccmt", help="which sources to use, e.g. 'ccmt'")
    a = ap.parse_args()
    steps = set(a.steps.split(","))
    existing = [f for d in NOT_CORN.iterdir() if d.is_dir() for f in list_images(d)]
    names = {f.name for f in existing}
    known = Known()
    for f in existing:
        known.add(phash(prepare(f.read_bytes())))
    need = a.target - len(existing)
    added, dupes, tilted = {}, 0, 0
    log(f"--- fill_not_corn: have {len(existing)}, target {a.target}{' DRY RUN' if a.dry_run else ''}")

    def keep(img, archived):
        """archived photos: full check (some sets held flipped/tilted copies). AgML rice photos
        are single leaves on white, where different leaves often sit within hash distance 4,
        so only near-exact copies (distance <= 2, unflipped) count as duplicates."""
        nonlocal dupes, tilted
        if archived and is_tilted_copy(img):
            tilted += 1
            return False
        hashes = variant_hashes(img)
        if known.seen(hashes) if archived else known.seen(hashes[:1], max_distance=2):
            dupes += 1
            return False
        known.add(hashes[0])
        return True

    # 1. archived rice/banana plant photos
    for f, dest, credit in (archive_candidates() if "archive" in steps else []):
        if sum(added.values()) >= need:
            break
        if f.name in names:
            dupes += 1
            continue
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        if not keep(img, True):
            continue
        added[dest] = added.get(dest, 0) + 1
        names.add(f.name)
        if not a.dry_run:
            (NOT_CORN / dest).mkdir(exist_ok=True)
            shutil.copy2(f, NOT_CORN / dest / f.name)
            append_csv(CREDITS_FILE, CREDIT_FIELDS, {"file": f"not_corn\\{dest}\\{f.name}", "source": credit["source"],
                                                     "original_name": credit["original_name"],
                                                     "licence": credit["licence"], "citation": credit["citation"]})
    log(f"  after archive: {added}")

    # 2. archived AgML rice parquet files
    rng = random.Random(42)
    for key, folder, pattern, mapping, citation in (AGML_SETS if "agml" in steps else []):
        for label, dest in mapping.items():
            src = ParquetSource(key, "../archive_non_corn/downloads/" + folder, pattern, [label], "CC BY 4.0", citation)
            src.desc = f"{folder}/{pattern} ['{label}']"
            for name, data in src.candidates(rng, 10 ** 6):
                if sum(added.values()) >= need:
                    break
                try:
                    img = prepare(data)
                except Exception:
                    continue
                if not keep(img, False):
                    continue
                stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(name).stem)[:40]
                fname = f"{key}_{stem}_{hashlib.sha1(data).hexdigest()[:6]}.jpg"
                if fname in names:
                    dupes += 1
                    continue
                names.add(fname)
                added[dest] = added.get(dest, 0) + 1
                if not a.dry_run:
                    img.save(NOT_CORN / dest / fname, quality=92)
                    append_csv(CREDITS_FILE, CREDIT_FIELDS, {"file": f"not_corn\\{dest}\\{fname}", "source": src.desc,
                                                             "original_name": name, "licence": "CC BY 4.0",
                                                             "citation": citation})
    # 3. CCMT cashew / cassava / tomato field photos
    for z, member, dest, cls in (ccmt_candidates(rng) if "ccmt" in steps else []):
        if sum(added.values()) >= need:
            break
        data = z.read(member)
        try:
            img = prepare(data)
        except Exception:
            continue
        if not keep(img, False):
            continue
        stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(member).stem)[:40]
        fname = f"ccmt_{dest.split('_')[0]}_{stem}_{hashlib.sha1(data).hexdigest()[:6]}.jpg"
        if fname in names:
            dupes += 1
            continue
        names.add(fname)
        added[dest] = added.get(dest, 0) + 1
        if not a.dry_run:
            (NOT_CORN / dest).mkdir(exist_ok=True)
            img.save(NOT_CORN / dest / fname, quality=92)
            append_csv(CREDITS_FILE, CREDIT_FIELDS, {"file": f"not_corn\\{dest}\\{fname}",
                                                     "source": f"kaggle:{CCMT_KAGGLE}/{cls}", "original_name": member,
                                                     "licence": "CC BY 4.0", "citation": CCMT_CITE})

    total = len(existing) + sum(added.values())
    log(f"not_corn: {'would add' if a.dry_run else 'added'} {sum(added.values())} {added}; skipped {dupes} duplicates, "
        f"{tilted} tilted copies -> {'would be' if a.dry_run else 'now'} {total}")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        raise
