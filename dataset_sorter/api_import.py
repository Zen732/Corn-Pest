"""Import photos from Roboflow Universe, Kaggle, GBIF and Hugging Face datasets.

Roboflow and Kaggle need your free API keys, read from dataset_sorter/roboflow_key.txt and
~/.kaggle/kaggle.json; they are never printed or logged. GBIF and Hugging Face need no key.
Each preset in PRESETS names one dataset, the class it fills and its credit.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/api_import.py rf-fall-armyworm --target 5000 --dry-run
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/api_import.py rf-fall-armyworm --target 5000
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/api_import.py kg-armyworms --target 5000

Duplicates are skipped, including flipped and turned copies that some datasets already
contain. Field-photo presets also skip tilted copies with black wedges along the border
(rotation/shear augmentation) and zoomed/shifted copies (CLIP similarity). Roboflow photos
showing an annotated insect are labeled bug_visible=yes; the rest are left for the labeling
tool. Every photo is credited in ml/dataset/dataset_credits.csv. Re-running only adds what's
still missing. Downloaded Roboflow exports are kept in dataset_sorter/downloads/roboflow/.
"""
import argparse
import base64
import hashlib
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from bulk_import import (CREDIT_FIELDS, CREDITS_FILE, DATASET_DIR, DOWNLOADS, LABELS_FILE, RAW_DIR, UA,
                         append_csv, list_images, log, phash, prepare)

SORTER_DIR = Path(__file__).resolve().parent
RF_KEY_FILE = SORTER_DIR / "roboflow_key.txt"
KAGGLE_FILE = Path(os.environ.get("USERPROFILE", Path.home())) / ".kaggle" / "kaggle.json"
DUPLICATE_DISTANCE = 4
DOCTOR_MAIZ = "daiv05/corn-leaf-diseases-pests-and-deficiencies"
DOCTOR_MAIZ_CITE = ("Doctor Maiz: Corn Leaf Diseases, Pests and Deficiencies (daiv05), Hugging Face, "
                    "https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies; "
                    "compiled from public datasets listed at https://github.com/daiv05/maize-doctor-classifier#datasets")

PRESETS = {
    # Field photos of corn with fall armyworm caterpillars and damage. Healthy-plant and
    # maize-streak photos are left out.
    "rf-fall-armyworm": dict(
        kind="roboflow", project="fall-armyworm/fall-armyworm", version=2, cls="corn/fall_armyworm",
        key="rfFAW", want=["fall-armyworm-larva", "fall-armyworm-larval-damage", "fall-armyworm-frass"],
        skip=["maize-streak-disease"], bug=["fall-armyworm-larva"], tilted_check=True, clip_dupes=0.97,
        licence="CC BY 4.0",
        citation="'fall armyworm' dataset (2024), fall armyworm workspace, Roboflow Universe, "
                 "https://universe.roboflow.com/fall-armyworm/fall-armyworm"),
    # Armyworm caterpillars (several armyworm species) collected from the web.
    "kg-armyworms": dict(
        kind="kaggle", dataset="tarundalal/dangerous-insects-dataset", folder="farm_insects/Armyworms",
        cls="corn/armyworm", key="kgFarmInsects", licence="CC0 1.0 (as stated on Kaggle)",
        citation="Dangerous Farm Insects Dataset, Kaggle user tarundalal, "
                 "https://www.kaggle.com/datasets/tarundalal/dangerous-insects-dataset"),
    # GBIF photos from sources other than iNaturalist (already downloaded directly).
    # Adult/egg/pupa records and museum specimens are skipped; CLIP keeps caterpillars and damage.
    "gbif-cutworm": dict(
        kind="gbif", taxon_key=5109882, species="Spodoptera litura", cls="corn/common_cutworm", key="gbif",
        clip_keep={"caterpillar", "damage"}, clip_dupes=0.97, licence="", citation=""),
    "gbif-armyworm": dict(
        kind="gbif", taxon_key=5802396, species="Mythimna separata", cls="corn/armyworm", key="gbif",
        clip_keep={"caterpillar", "damage"}, clip_dupes=0.97, licence="", citation=""),
    # Doctor Maiz (El Salvador): field photos of healthy and diseased corn, streamed from
    # Hugging Face .tar shards. `take` = share of each folder to sample (spreads the picks
    # over the dataset's sources); shards = the ones that hold those folders. Close-up leaf
    # photos look alike, so the CLIP near-copy cutoff is 0.985 here (0.97 skipped almost
    # half of the healthy photos); the closest kept pairs are checked by eye afterwards.
    "hf-doctor-maiz-healthy": dict(
        kind="hf_tar", repo=DOCTOR_MAIZ, shards=[7, 8, 9, 10, 11, 12], take={"healthy/real/": 1.0},
        cls="corn/healthy", key="dmz", label="no", tilted_check=True, clip_dupes=0.985,
        licence="CC BY-NC-SA 4.0", citation=DOCTOR_MAIZ_CITE),
    "hf-doctor-maiz-damage": dict(
        kind="hf_tar", repo=DOCTOR_MAIZ, shards=[0, 4, 5, 6, 7, 13, 15, 17, 21, 22],
        take={"common_rust/real/": 1.0, "gray_leaf_spot/real/": 1.0, "lethal_necrosis/real/": 1.0,
              "northern_corn_leaf_blight/real/": 1.0},
        cls="corn/unidentified_damage", key="dmz", label="no", tilted_check=True, clip_dupes=0.985,
        licence="CC BY-NC-SA 4.0", citation=DOCTOR_MAIZ_CITE),
}
GBIF_SKIP_DATASETS = {"50c9509d-22c7-4a22-a47d-8c48425ef4a7"}  # iNaturalist research grade
GBIF_SKIP_LIFE_STAGES = {"Adult", "Imago", "Egg", "Pupa"}
GBIF_SKIP_BASIS = {"PRESERVED_SPECIMEN", "MATERIAL_SAMPLE"}


# ---------------------------------------------------------------- web helpers

def request(url, headers=None, timeout=120):
    """Open a URL, waiting and retrying on 'too many requests' (429) and network errors."""
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    for attempt in range(6):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(30 * (attempt + 1))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(10)


def kaggle_auth():
    k = json.loads(KAGGLE_FILE.read_text())
    return {"Authorization": "Basic " + base64.b64encode(f"{k['username']}:{k['key']}".encode()).decode()}


def rf_key():
    return RF_KEY_FILE.read_text().strip()


# ---------------------------------------------------------------- sources
# Each yields (original_name, image_bytes, bug_visible) - bug_visible is "yes" or "".

def roboflow_candidates(p, rng):
    zpath = DOWNLOADS / "roboflow" / f"{p['project'].replace('/', '__')}__v{p['version']}_coco.zip"
    if not zpath.exists():
        link = None
        for _ in range(30):  # Roboflow may need a minute to prepare the export
            with request(f"https://api.roboflow.com/{p['project']}/{p['version']}/coco?api_key={rf_key()}") as r:
                link = json.load(r).get("export", {}).get("link")
            if link:
                break
            time.sleep(10)
        if not link:
            raise RuntimeError("Roboflow export not ready")
        zpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = zpath.with_suffix(".part")
        with request(link, timeout=600) as r, open(tmp, "wb") as f:
            size = int(r.headers.get("Content-Length") or 0)
            with tqdm(total=size, unit="B", unit_scale=True, desc="download export") as bar:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(zpath)
    z = zipfile.ZipFile(zpath)
    want, skip, bug = set(p["want"]), set(p["skip"]), set(p["bug"])
    with_bug, without = [], []
    for ann in [n for n in z.namelist() if n.endswith("_annotations.coco.json")]:
        coco = json.loads(z.read(ann))
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        per_image = {}
        for a in coco["annotations"]:
            per_image.setdefault(a["image_id"], set()).add(cat.get(a["category_id"], ""))
        folder = ann.rsplit("/", 1)[0]
        for im in coco["images"]:
            names = per_image.get(im["id"], set())
            if names & want and not names & skip:
                (with_bug if names & bug else without).append(f"{folder}/{im['file_name']}")
    rng.shuffle(with_bug)
    rng.shuffle(without)
    log(f"  roboflow {p['project']} v{p['version']}: {len(with_bug)} photos with an insect, {len(without)} damage only")
    for path in with_bug:
        yield path.rsplit("/", 1)[-1], z.read(path), "yes"
    for path in without:
        yield path.rsplit("/", 1)[-1], z.read(path), ""


def kaggle_candidates(p, rng):
    auth = kaggle_auth()
    names, token = [], ""
    while True:
        q = urllib.parse.urlencode({"pageSize": 200, **({"pageToken": token} if token else {})})
        with request(f"https://www.kaggle.com/api/v1/datasets/list/{p['dataset']}?{q}", auth) as r:
            data = json.load(r)
        names += [f["name"] for f in data.get("datasetFiles") or []
                  if f["name"].startswith(p["folder"] + "/") and Path(f["name"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        token = data.get("nextPageToken") or ""
        if not token:
            break
    rng.shuffle(names)
    log(f"  kaggle {p['dataset']}/{p['folder']}: {len(names)} files")
    for name in names:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{p['dataset']}?{urllib.parse.urlencode({'fileName': name})}"
        try:
            with request(url, auth) as r:
                yield name.rsplit("/", 1)[-1], r.read(), ""
        except Exception:
            yield name.rsplit("/", 1)[-1], b"", ""
        time.sleep(0.5)


class ResumableHTTP:
    """File-like download that reconnects with an HTTP Range request when the connection drops.

    Long .tar streams pause while photos are being checked, and the server may close a
    paused connection; this picks the download up at the same byte instead of failing.
    """

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.resp = request(url, timeout=120)
        self.size = int(self.resp.headers.get("Content-Length") or 0)

    def read(self, n=-1):
        for attempt in range(8):
            try:
                b = self.resp.read(n)
                if not b and n != 0 and self.size and self.pos < self.size:
                    raise IOError("connection closed early")  # server hung up quietly: resume
                self.pos += len(b)
                return b
            except Exception:
                time.sleep(5 * (attempt + 1))
                try:
                    self.resp.close()
                except Exception:
                    pass
                self.resp = request(self.url, {"Range": f"bytes={self.pos}-"}, timeout=120)
        raise IOError(f"download kept failing at byte {self.pos}")

    def close(self):
        self.resp.close()


def hf_tar_candidates(p, rng):
    """Stream several Hugging Face .tar shards at once; yield photos from the `take` folders."""
    import queue
    import tarfile
    import threading
    q, stop = queue.Queue(maxsize=256), threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=1)
                return
            except queue.Full:
                pass

    def worker(shard):
        pick = random.Random(f"{p['repo']}-{shard}")
        url = f"https://huggingface.co/datasets/{p['repo']}/resolve/main/clean-{shard:05d}.tar"
        src = None
        try:
            src = ResumableHTTP(url)
            with tarfile.open(fileobj=src, mode="r|") as tar:
                for m in tar:
                    if stop.is_set():
                        return
                    share = next((v for k, v in p["take"].items() if m.name.startswith(k)), 0)
                    if m.isfile() and share and pick.random() < share:
                        put((m.name, tar.extractfile(m).read()))
        except Exception as e:
            log(f"  shard {shard} stopped early: {type(e).__name__}: {str(e)[:80]}")
        finally:
            if src:
                src.close()
            put(None)

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in p["shards"]]
    for t in threads:
        t.start()
    finished = 0
    try:
        while finished < len(threads):
            item = q.get()
            if item is None:
                finished += 1
                continue
            yield item[0].rsplit("/", 1)[-1], item[1], p.get("label", ""), {"original": item[0]}
    finally:
        stop.set()


def licence_name(text):
    """Open licence name from a GBIF licence URL or code; '' for no-derivatives / all rights reserved."""
    u = (text or "").lower().replace("_", "-").replace(" ", "-")
    if "-nd" in u or "rights-reserved" in u:
        return ""
    for key, name in (("publicdomain/zero", "CC0 1.0"), ("cc0", "CC0 1.0"), ("by-nc-sa", "CC BY-NC-SA 4.0"),
                      ("by-nc", "CC BY-NC 4.0"), ("by-sa", "CC BY-SA 4.0"), ("/by/", "CC BY 4.0"), ("cc-by-4", "CC BY 4.0")):
        if key in u:
            return name
    return ""


def gbif_candidates(p, rng):
    """GBIF photo records of one species from every source except those in GBIF_SKIP_DATASETS."""
    base = "https://api.gbif.org/v1/occurrence/search"
    q = {"taxonKey": p["taxon_key"], "mediaType": "StillImage", "limit": 0, "facet": "datasetKey", "facetLimit": 100}
    with request(f"{base}?{urllib.parse.urlencode(q)}") as r:
        facets = json.load(r)["facets"]
    datasets = [c["name"] for c in (facets[0]["counts"] if facets else []) if c["name"] not in GBIF_SKIP_DATASETS]
    for ds in datasets:
        with request(f"https://api.gbif.org/v1/dataset/{ds}") as r:
            title = json.load(r).get("title", ds)
        offset = 0
        while True:
            q = {"taxonKey": p["taxon_key"], "datasetKey": ds, "mediaType": "StillImage", "limit": 300, "offset": offset}
            with request(f"{base}?{urllib.parse.urlencode(q)}") as r:
                page = json.load(r)
            for o in page["results"]:
                if (o.get("lifeStage") or "") in GBIF_SKIP_LIFE_STAGES or o.get("basisOfRecord") in GBIF_SKIP_BASIS:
                    continue
                media = next((m for m in o.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")), None)
                lic = licence_name((media or {}).get("license") or o.get("license"))
                if not media or not lic:
                    continue
                try:
                    with urllib.request.urlopen(urllib.request.Request(media["identifier"], headers=UA), timeout=40) as r:
                        data = r.read()
                except Exception:
                    data = b""
                who = media.get("rightsHolder") or media.get("creator") or o.get("rightsHolder") or o.get("recordedBy") or "unknown"
                who = re.sub(r"<[^>]+>", "", who).strip() or "unknown"  # some sources send HTML links
                url = f"https://www.gbif.org/occurrence/{o['key']}"
                yield str(o["key"]), data, "", {"licence": lic, "source": f"gbif:{ds}", "original": url,
                                                 "citation": f"{who}, {title} ({p['species']}), via GBIF.org, {url}"}
            offset += 300
            if page.get("endOfRecords", True):
                break


# ---------------------------------------------------------------- duplicate checks

def is_tilted_copy(img):
    """True for rotated/sheared copies: pure-black wedges along the photo border.

    Only used for field-photo presets (tilted_check=True); a dark-background insect
    photo would also trip it.
    """
    a = np.asarray(img.convert("L").resize((128, 128)))
    frame = np.concatenate([a[:2].ravel(), a[-2:].ravel(), a[:, :2].ravel(), a[:, -2:].ravel()])
    return (frame < 12).mean() > 0.02


class ClipDupes:
    """Catches zoomed / shifted copies of a photo that perceptual hashes miss (CLIP similarity)."""

    def __init__(self, threshold, existing, judge):
        import torch
        self.torch, self.t = torch, threshold
        self.j = judge
        self.e = np.zeros((0, 512), dtype=np.float32)
        for i in range(0, len(existing), 64):
            self.e = np.vstack([self.e, self.embed([Image.open(f).convert("RGB") for f in existing[i:i + 64]])])

    def embed(self, ims):
        with self.torch.no_grad():
            x = self.torch.stack([self.j.pre(im) for im in ims]).to(self.j.device)
            e = self.j.model.encode_image(x).float()
        return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def seen_or_add(self, img):
        e = self.embed([img])
        if len(self.e) and float((self.e @ e[0]).max()) > self.t:
            return True
        self.e = np.vstack([self.e, e])
        return False


def variant_hashes(img):
    """Perceptual hashes of the photo and its flipped / turned copies."""
    T = Image.Transpose
    return [phash(img)] + [phash(img.transpose(t)) for t in
                           (T.FLIP_LEFT_RIGHT, T.FLIP_TOP_BOTTOM, T.ROTATE_90, T.ROTATE_180, T.ROTATE_270)]


class Known:
    """Fast near-duplicate lookup over 64-bit perceptual hashes."""

    def __init__(self):
        self.h = np.zeros(0, dtype=np.uint64)

    def add(self, h):
        self.h = np.append(self.h, np.uint64(h))

    def seen(self, hashes, max_distance=DUPLICATE_DISTANCE):
        if not len(self.h):
            return False
        for h in hashes:
            x = np.bitwise_xor(self.h, np.uint64(h))
            if np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1).min() <= max_distance:
                return True
        return False


# ---------------------------------------------------------------- import

def run(name, target, dry_run):
    p = PRESETS[name]
    crop, pest = p["cls"].split("/")
    folder = RAW_DIR / p["cls"]
    existing = list_images(folder)
    need = target - len(existing)
    if need <= 0:
        log(f"{p['cls']}: already {len(existing)}, nothing to do")
        return
    known = Known()
    for f in existing + list_images(DATASET_DIR / "rejected" / p["cls"]) + list_images(DATASET_DIR / "reserve" / p["cls"]):
        try:
            known.add(phash(prepare(f.read_bytes())))
        except Exception:
            pass
    judge = None
    if p.get("clip_dupes") or p.get("clip_keep"):
        from inat_larvae import Judge
        judge = Judge()
    # CLIP check compares against photos already imported from this same dataset.
    clip = ClipDupes(p["clip_dupes"], [f for f in existing if f.name.startswith(p["key"] + "_")], judge) \
        if p.get("clip_dupes") else None
    sources = {"roboflow": roboflow_candidates, "kaggle": kaggle_candidates,
               "hf_tar": hf_tar_candidates, "gbif": gbif_candidates}
    source = sources[p["kind"]](p, random.Random(42))
    added = {}
    tilted = dupes = failed = off_topic = 0
    log(f"--- {name}: filling {p['cls']} (have {len(existing)}, target {target}){' DRY RUN' if dry_run else ''}")
    bar = tqdm(total=need, desc=f"{p['cls']:<22}", unit="photo", dynamic_ncols=True)
    for orig, data, bug, *rest in source:
        credit = rest[0] if rest else {}
        if sum(added.values()) >= need:
            break
        try:
            img = prepare(data)
        except Exception:
            failed += 1
            continue
        if p.get("tilted_check") and is_tilted_copy(img):
            tilted += 1
            continue
        hashes = variant_hashes(img)
        if known.seen(hashes):
            dupes += 1
            continue
        if p.get("clip_keep"):  # e.g. keep caterpillars and damage, drop adult moths
            group = judge.groups([img])[0]
            if group not in p["clip_keep"]:
                off_topic += 1
                continue
            bug = "yes" if group == "caterpillar" else ""
        if clip and clip.seen_or_add(img):
            dupes += 1
            continue
        known.add(hashes[0])
        added[bug] = added.get(bug, 0) + 1
        bar.update(1)
        bar.set_postfix(added=sum(added.values()), off_topic=off_topic, tilted=tilted, duplicates=dupes)
        if dry_run:
            continue
        stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(orig).stem)[:40]
        dest = folder / f"{p['key']}_{stem}_{hashlib.sha1(data).hexdigest()[:6]}.jpg"
        img.save(dest, quality=92)
        rel = str(dest.relative_to(RAW_DIR))
        if bug:
            append_csv(LABELS_FILE, ["path", "crop", "pest_class", "bug_visible"],
                       {"path": rel, "crop": crop, "pest_class": pest, "bug_visible": bug})
        src = credit.get("source") or {"roboflow": lambda: f"roboflow:{p['project']}/v{p['version']}",
                                       "kaggle": lambda: f"kaggle:{p['dataset']}/{p['folder']}",
                                       "hf_tar": lambda: f"huggingface:{p['repo']}"}[p["kind"]]()
        append_csv(CREDITS_FILE, CREDIT_FIELDS, {"file": rel, "source": src, "original_name": credit.get("original", orig),
                                                 "licence": credit.get("licence") or p["licence"],
                                                 "citation": credit.get("citation") or p["citation"]})
    source.close()  # stops background downloads (hf_tar) once the target is reached
    bar.close()
    total = len(existing) + sum(added.values())
    note = "" if total >= target else "  (source ran out)"
    verb = "would add" if dry_run else "added"
    labels = ", ".join(f"{n} {'unlabeled' if k == '' else 'bug_visible=' + k}" for k, n in sorted(added.items())) or "0"
    log(f"{p['cls']}: {verb} {sum(added.values())} ({labels}); skipped {off_topic} off-topic (e.g. moths), "
        f"{tilted} tilted copies, {dupes} duplicates, {failed} unreadable -> {'would be' if dry_run else 'now'} {total}{note}")
    log("DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true", help="count what would be added, save nothing")
    a = ap.parse_args()
    run(a.preset, a.target, a.dry_run)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        raise
