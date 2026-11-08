"""Pre-label unlabeled photos as bug_visible yes/no with CLIP, leaving unsure ones for you.

CLIP is weak at telling insect species apart, but decent at "is there an insect in this
photo?". In practice its "yes" is reliable and its "no" is not (it misses small larvae and
planthoppers), so --apply only writes the labels it can be trusted with (see final_label).
Everything else stays unlabeled, so ml/src/label_bug_visible.py shows it to you as usual.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/prelabel_bug_visible.py           # check only
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/prelabel_bug_visible.py --apply   # write labels

Check-only writes dataset_sorter/audit/prelabel_log.csv and contact sheets
(prelabel_yes.jpg, prelabel_no.jpg, prelabel_unsure.jpg). --apply adds the yes/no rows to
ml/dataset/labels.csv and keeps the scores in ml/dataset/prelabel_log.csv, so any CLIP
label can be traced and undone.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageOps

from sort_images import embed_images, load_image

SORTER_DIR = Path(__file__).resolve().parent
DATASET_DIR = SORTER_DIR.parent / "ml" / "dataset"
RAW_DIR = DATASET_DIR / "raw"
LABELS_FILE = DATASET_DIR / "labels.csv"
AUDIT_DIR = SORTER_DIR / "audit"
APPLIED_LOG = DATASET_DIR / "prelabel_log.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BUG = ["a close-up photo of an insect", "a caterpillar on a plant", "a moth",
       "a small bug on a leaf or stem", "a planthopper insect",
       "an insect larva inside a cut plant stem", "a cluster of insect eggs on a leaf"]
NO_BUG = ["a damaged plant with no insects visible", "corn leaves with ragged holes and no insect",
          "a corn stalk with holes and no insect", "a corn ear with damaged kernels and no insect",
          "a diseased leaf with spots", "a healthy green leaf", "a field of corn with dry brown leaves"]
OTHER = ["a diagram or chart with text", "a collage of several photos"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="write the confident labels to labels.csv")
    p.add_argument("--sure", type=float, default=0.85,
                   help="label yes when the bug share is at least this, no when at most 1 - this (default 0.85)")
    return p.parse_args()


def unlabeled_photos():
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        done = {r["path"] for r in csv.DictReader(f)}
    photos = [p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(p for p in photos if str(p.relative_to(RAW_DIR)) not in done)


def sheet(items, title, out_path, n=40, per_row=10, tile=125):
    items = random.Random(0).sample(items, min(n, len(items)))
    rows = max(1, -(-len(items) // per_row))
    img = Image.new("RGB", (per_row * tile, 26 + rows * tile), "white")
    d = ImageDraw.Draw(img)
    d.text((6, 6), title, fill="black")
    for k, (path, share, cls) in enumerate(items):
        x, y = (k % per_row) * tile, 26 + (k // per_row) * tile
        try:
            img.paste(ImageOps.fit(load_image(path), (tile - 4, tile - 4)), (x + 2, y + 2))
        except Exception:
            continue
        d.rectangle((x + 2, y + 2, x + 120, y + 15), fill="black")
        d.text((x + 4, y + 3), f"{share:.2f} {cls.split('/')[-1][:14]}", fill="white")
    img.save(out_path, quality=88)


def main():
    args = parse_args()
    photos = unlabeled_photos()
    print(f"{len(photos)} unlabeled photo(s)")
    if not photos:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = BUG + NO_BUG + OTHER
    with torch.no_grad():
        text = model.encode_text(tokenizer([f"a photo of {p}" for p in prompts]).to(device))
        text = torch.nn.functional.normalize(text.float(), dim=-1).cpu().numpy()
    scale = model.logit_scale.exp().item()

    emb, _ = embed_images(photos, model, preprocess, device, batch_size=64, workers=0)
    results = {"yes": [], "no": [], "unsure": []}
    log_rows = []
    nb, nn = len(BUG), len(NO_BUG)
    for path, e in zip(photos, emb):
        if e is None:
            continue
        logits = scale * (text @ e)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        p_bug, p_no, p_other = p[:nb].sum(), p[nb:nb + nn].sum(), p[nb + nn:].sum()
        share = float(p_bug / (p_bug + p_no))
        if p_other > 0.5:
            decision = "unsure"
        elif share >= args.sure:
            decision = "yes"
        elif share <= 1 - args.sure:
            decision = "no"
        else:
            decision = "unsure"
        rel = path.relative_to(RAW_DIR)
        cls = "/".join(rel.parts[:2])
        results[decision].append((path, share, cls))
        log_rows.append({"path": str(rel), "class": cls, "bug_share": f"{share:.3f}",
                         "other": f"{p_other:.3f}", "best_match": prompts[int(p.argmax())], "decision": decision})

    print(f"\n{'class':<28}{'yes':>6}{'no':>6}{'unsure':>8}")
    for cls in sorted({r["class"] for r in log_rows}):
        c = {d: sum(1 for r in log_rows if r["class"] == cls and r["decision"] == d) for d in results}
        print(f"{cls:<28}{c['yes']:>6}{c['no']:>6}{c['unsure']:>8}")

    AUDIT_DIR.mkdir(exist_ok=True)
    for decision, items in results.items():
        sheet(items, f"prelabel {decision}: {len(items)} photos (random 40; number = bug share)",
              AUDIT_DIR / f"prelabel_{decision}.jpg")
    with open(AUDIT_DIR / "prelabel_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)

    if not args.apply:
        print(f"\nCheck only. Contact sheets and log in {AUDIT_DIR}. Re-run with --apply to write labels.")
        return
    # Which CLIP calls to trust, from checking the contact sheets (2026-10-14):
    # its "yes" was right ~39/40, but its "no" missed ~1 in 6 small larvae / planthoppers.
    # So in pest folders only "yes" is written; in unidentified_damage (damage-only by
    # definition) everything is "no" except photos CLIP thinks show a bug, which you review.
    def final_label(r):
        if r["class"].endswith("/unidentified_damage"):
            return None if r["decision"] == "yes" else "no"
        return "yes" if r["decision"] == "yes" else None

    applied = []
    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "crop", "pest_class", "bug_visible"])
        for r in log_rows:
            label = final_label(r)
            if label:
                crop, pest = r["class"].split("/")
                w.writerow({"path": r["path"], "crop": crop, "pest_class": pest, "bug_visible": label})
                applied.append({**r, "decision": label})
    new = not APPLIED_LOG.exists()
    with open(APPLIED_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        if new:
            w.writeheader()
        w.writerows(applied)
    print(f"\nWrote {len(applied)} label(s). {len(log_rows) - len(applied)} photo(s) left for label_bug_visible.py.")


if __name__ == "__main__":
    main()
