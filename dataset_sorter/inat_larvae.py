"""Download iNaturalist photos of a caterpillar pest, keeping only caterpillars and corn damage.

Most iNaturalist photos of armyworms and cutworms are adult moths at night lights, which
farmers rarely photograph. This downloader checks every photo with CLIP as it arrives and
keeps only those that show a caterpillar or a damaged corn plant (see CLIP_GROUPS).

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/inat_larvae.py --class corn/fall_armyworm --target 5000

Sources are tried in order: research grade (confirmed by two identifiers), then "needs ID"
(not yet confirmed - credited as such; check a sample by eye). Kept photos go to
ml/dataset/raw/<class>/ and ml/dataset/inaturalist_credits.csv; caterpillar photos are
labeled bug_visible=yes, damage photos are left for the labeling tool. Rejected
observations are listed in dataset_sorter/inat_rejected.csv so they are never fetched again.
"""
import argparse
import csv
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import torch
from tqdm import tqdm

from inat_download import (API, CREDIT_FIELDS, CREDITS_FILE, LABELS_FILE, LARVA, LICENSES, RAW_DIR,
                           append_csv, fetch_photo, first_open_photo, get_json, is_duplicate, list_images,
                           log, phash)

REJECTED_FILE = Path(__file__).resolve().parent / "inat_rejected.csv"
SPECIES = {
    "corn/fall_armyworm": "Spodoptera frugiperda",
    "corn/common_cutworm": "Spodoptera litura",
    "corn/armyworm": "Mythimna separata",
}
CLIP_GROUPS = {
    "caterpillar": ["a photo of a caterpillar", "a caterpillar on a leaf", "a caterpillar on a hand"],
    "damage": ["a corn plant with damaged leaves", "a field of young corn plants"],
    "moth": ["a photo of a moth", "a moth resting on a wall"],
    "other": ["a pupa", "insect eggs on a leaf", "a diagram with text"],
}
KEEP = {"caterpillar", "damage"}
CHUNK = 200  # photos judged per batch (one API page)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class", dest="cls", required=True, choices=sorted(SPECIES))
    p.add_argument("--target", type=int, default=5000)
    p.add_argument("--workers", type=int, default=32)
    return p.parse_args()


def seen_ids():
    ids = set()
    for path in (CREDITS_FILE, REJECTED_FILE):
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                ids |= {r["observation_id"] for r in csv.DictReader(f)}
    return ids


class Judge:
    """CLIP: which group does a photo belong to?"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.model = self.model.to(self.device).eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        prompts = [p for ps in CLIP_GROUPS.values() for p in ps]
        self.owner = [g for g, ps in CLIP_GROUPS.items() for _ in ps]
        with torch.no_grad():
            t = self.model.encode_text(tok(prompts).to(self.device))
        self.text = torch.nn.functional.normalize(t.float(), dim=-1)
        self.scale = self.model.logit_scale.exp().item()

    def groups(self, images):
        with torch.no_grad():
            x = torch.stack([self.pre(im) for im in images]).to(self.device)
            e = torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1)
            p = (self.scale * e @ self.text.T).softmax(dim=-1).cpu().numpy()
        out = []
        for row in p:
            score = {}
            for g, v in zip(self.owner, row):
                score[g] = score.get(g, 0.0) + float(v)
            out.append(max(score, key=score.get))
        return out


def main():
    args = parse_args()
    crop, pest = args.cls.split("/")
    folder = RAW_DIR / args.cls
    existing = list_images(folder)
    need = args.target - len(existing)
    if need <= 0:
        log(f"{args.cls}: already {len(existing)}, nothing to do")
        return
    known = []
    for f in existing:
        try:
            from PIL import Image, ImageOps
            known.append(phash(ImageOps.exif_transpose(Image.open(f)).convert("RGB")))
        except Exception:
            pass
    done = seen_ids()
    judge = Judge()
    species = SPECIES[args.cls]
    added = {"caterpillar": 0, "damage": 0}
    rejected, dupes = 0, 0
    bar = tqdm(total=need, desc=f"{args.cls:<22}", unit="photo", dynamic_ncols=True)
    log(f"--- {args.cls}: keeping caterpillars + damage, target {args.target}, have {len(existing)}")
    with ThreadPoolExecutor(args.workers) as pool:
        for grade, extra in (("research", LARVA), ("research", {}), ("needs_id", {})):
            id_above = 0
            while sum(added.values()) < need:
                q = {"taxon_name": species, "quality_grade": grade, "photos": "true",
                     "photo_license": ",".join(sorted(LICENSES)), "order_by": "id", "order": "asc",
                     "id_above": id_above, "per_page": 200, **extra}
                page = get_json(f"{API}?{urllib.parse.urlencode(q)}")["results"]
                time.sleep(1)
                if not page:
                    break
                id_above = page[-1]["id"]
                todo = [(o, first_open_photo(o)) for o in page if str(o["id"]) not in done]
                todo = [(o, ph) for o, ph in todo if ph is not None]
                for i in range(0, len(todo), CHUNK):
                    chunk = todo[i:i + CHUNK]
                    # Judge from the small preview (CLIP only needs 224 px); fetch full size only for keepers.
                    previews = list(pool.map(fetch_photo, [ph["url"].replace("/square.", "/small.") for _, ph in chunk]))
                    ok = [(o, ph, r) for (o, ph), r in zip(chunk, previews) if r is not None]
                    if not ok:
                        continue
                    groups = judge.groups([r[0] for _, _, r in ok])
                    keepers = []
                    for (obs, photo, _), group in zip(ok, groups):
                        oid = str(obs["id"])
                        done.add(oid)
                        if group in KEEP:
                            keepers.append((obs, photo, group))
                        else:
                            append_csv(REJECTED_FILE, ["observation_id", "class", "group"],
                                       {"observation_id": oid, "class": args.cls, "group": group})
                            rejected += 1
                    full = list(pool.map(fetch_photo, [ph["url"].replace("/square.", "/medium.") for _, ph, _ in keepers]))
                    for (obs, photo, group), result in zip(keepers, full):
                        if result is None:
                            continue
                        img, h = result
                        oid = str(obs["id"])
                        if is_duplicate(h, known):
                            dupes += 1
                            continue
                        if sum(added.values()) >= need:
                            break
                        known.append(h)
                        dest = folder / f"inat_{oid}.jpg"
                        img.save(dest, quality=92)
                        rel = str(dest.relative_to(RAW_DIR))
                        if group == "caterpillar":
                            append_csv(LABELS_FILE, ["path", "crop", "pest_class", "bug_visible"],
                                       {"path": rel, "crop": crop, "pest_class": pest, "bug_visible": "yes"})
                        append_csv(CREDITS_FILE, CREDIT_FIELDS, {
                            "file": rel, "observation_id": oid,
                            "observation_url": f"https://www.inaturalist.org/observations/{oid}",
                            "species": f"{species} ({grade})", "license": photo["license_code"],
                            "attribution": photo.get("attribution", ""),
                            "observed_on": obs.get("observed_on") or "", "place": obs.get("place_guess") or ""})
                        added[group] += 1
                        bar.update(1)
                        bar.set_postfix(caterpillars=added["caterpillar"], damage=added["damage"], rejected=rejected)
            if sum(added.values()) >= need:
                break
    bar.close()
    total = len(existing) + sum(added.values())
    note = "" if total >= args.target else "  (no more photos available on iNaturalist)"
    log(f"{args.cls}: added {added['caterpillar']} caterpillars + {added['damage']} damage, rejected {rejected} "
        f"(moths/other), skipped {dupes} duplicates -> now {total}{note}")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FAILED:\n" + traceback.format_exc())
        raise
