"""Version 2: close-ups of small things that are not corn (grains, beans, pebbles, shells), for the
corn check. Why: a grain of rice filling the photo passed the corn check and was shown as "Insect
not recognized" (23 Nov 2026). The corn check accepts close-ups of the five pests (farmers
photograph them up close), and had never seen close-ups of small objects that are not insects.

Adds ml/dataset/not_corn/small_objects/ (nothing existing is changed). Kaggle datasets with open
licences, single files downloaded with the user's Kaggle token (~/.kaggle/kaggle.json, never
printed); photos spread evenly over each dataset's folders:

    rice      600  muratkokludataset/rice-image-dataset (CC0), 120 of each of 5 varieties
    coffee    400  gpiosenka/coffee-bean-dataset-resized-224-x-224 (CC BY-SA 4.0), 100 per roast
    pebbles   350  vencerlanz09/shells-or-pebbles-an-image-classification-dataset (CC BY-SA 4.0)
    shells    250  same dataset
    (wheat: muratkokludataset/durum-wheat-dataset was tried and left out: its .tif video frames
    did not open after a single-file download)

Credited in ml/dataset/dataset_credits.csv (backed up first). Safe to run again: it tops up.
Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/fill_small_objects.py
"""
import json
import random
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

from api_import import kaggle_auth, request
from bulk_import import CREDIT_FIELDS, CREDITS_FILE, DATASET_DIR, append_csv, list_images, phash, prepare
from fill_real_world import backup
from inat_download import is_duplicate

OUT = DATASET_DIR / "not_corn" / "small_objects"
LISTS = Path(__file__).resolve().parent / "downloads" / "kaggle_small_objects"
RICE_CITE = ("Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning "
             "methods. Computers and Electronics in Agriculture, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285")
WHEAT_CITE = ("Kaya, E., & Saritas, I. (2019). Towards a real-time sorting system: identification of vitreous durum "
              "wheat kernels using ANN based on their morphological, colour, wavelet and gaborlet features. Computers "
              "and Electronics in Agriculture, 166, 105016. https://doi.org/10.1016/j.compag.2019.105016")
COFFEE_CITE = ("Ontoum, S., Khemanantakul, T., Sroison, P., Triyason, T., & Watanapa, B. (2022). Coffee Roast "
               "Intelligence. arXiv:2206.01841")
PEBBLE_CITE = ("Kaggle dataset by vencerlanz09, 'Shells or Pebbles: An Image Classification Dataset' "
               "(the photos' original source is not stated)")
# name, Kaggle dataset, folders to use, photos per folder, licence, citation
SOURCES = [
    ("rice", "muratkokludataset/rice-image-dataset",
     [f"Rice_Image_Dataset/{v}" for v in ("Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag")], 120,
     "CC0: Public Domain", RICE_CITE),
    ("coffee", "gpiosenka/coffee-bean-dataset-resized-224-x-224",
     [f"train/{v}" for v in ("Dark", "Green", "Light", "Medium")], 100, "CC BY-SA 4.0", COFFEE_CITE),
    ("pebbles", "vencerlanz09/shells-or-pebbles-an-image-classification-dataset", ["Pebbles"], 350,
     "CC BY-SA 4.0", PEBBLE_CITE),
    ("shells", "vencerlanz09/shells-or-pebbles-an-image-classification-dataset", ["Shells"], 250,
     "CC BY-SA 4.0", PEBBLE_CITE),
]
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def file_list(ref, auth):
    """All file names of a Kaggle dataset (cached in downloads/)."""
    f = LISTS / (ref.replace("/", "__") + ".json")
    if f.exists():
        return [n for n, _ in json.loads(f.read_text(encoding="utf-8"))]
    names, token = [], ""
    while True:
        q = urllib.parse.urlencode({"pageSize": 200, **({"pageToken": token} if token else {})})
        with request(f"https://www.kaggle.com/api/v1/datasets/list/{ref}?{q}", auth) as r:
            data = json.load(r)
        names += [(x["name"], x.get("totalBytes")) for x in data.get("datasetFiles") or []]
        token = data.get("nextPageToken") or ""
        if not token:
            break
    LISTS.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(names), encoding="utf-8")
    return [n for n, _ in names]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    backup(CREDITS_FILE)
    auth = kaggle_auth()
    known = []
    for f in list_images(OUT):
        known.append(phash(prepare(f.read_bytes())))
    for name, ref, folders, per_folder, licence, cite in SOURCES:
        by_folder = defaultdict(list)
        for n in file_list(ref, auth):
            folder = n.rsplit("/", 1)[0] if "/" in n else ""
            if folder in folders and n.lower().endswith(IMAGE_EXT):
                by_folder[folder].append(n)
        for folder in folders:
            tag = f"{name}_{folders.index(folder) + 1}"
            have = sum(1 for f in list_images(OUT) if f.name.startswith(tag + "_"))
            files = sorted(by_folder[folder])
            random.Random(f"small_objects:{folder}").shuffle(files)
            added = 0
            for n in files:
                if have + added >= per_folder:
                    break
                orig = n.rsplit("/", 1)[-1]
                fname = f"{tag}_{Path(orig).stem.replace(' ', '_')}.jpg"
                if (OUT / fname).exists():
                    continue
                url = (f"https://www.kaggle.com/api/v1/datasets/download/{ref}?"
                       + urllib.parse.urlencode({"fileName": n}))
                try:
                    with request(url, auth) as r:
                        img = prepare(r.read())
                except Exception as e:
                    print(f"  skip {n}: {e}")
                    continue
                h = phash(img)
                if is_duplicate(h, known):
                    continue
                known.append(h)
                img.save(OUT / fname, quality=92)
                append_csv(CREDITS_FILE, CREDIT_FIELDS, {
                    "file": f"not_corn\\small_objects\\{fname}", "source": f"kaggle:{ref}/{folder}",
                    "original_name": orig, "licence": licence, "citation": cite})
                added += 1
                time.sleep(0.3)
            print(f"{tag} ({folder.rsplit('/', 1)[-1]}): {have + added} photos", flush=True)
    print(f"small_objects: {len(list_images(OUT))} photos")


if __name__ == "__main__":
    main()
