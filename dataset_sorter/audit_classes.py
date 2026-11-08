"""Flag photos in ml/dataset/raw/ pest folders that don't show what that pest class needs.

CLIP compares each photo to short text descriptions: what a correct photo of that class
shows ("keep" prompts) and the usual wrong photos ("reject" prompts). The share of the
match that goes to the keep prompts is the photo's keep score (0-1). A photo is flagged
when its keep score is below that class's cutoff in CLASS_THRESHOLDS.

CLIP cannot tell insect species apart (one stink bug from another). It catches the
obvious mistakes - a moth in the planthopper folder, a longhorn beetle in the weevil
folder, a leafhopper where a caterpillar should be.

Run from the project root, with the sorter's environment:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/audit_classes.py           # check only
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/audit_classes.py --apply   # move flagged

Check-only writes dataset_sorter/audit/audit_log.csv and a contact sheet per class.
--apply MOVES flagged photos to ml/dataset/rejected/<crop>/<class>/ (nothing is deleted).
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw

from sort_images import embed_images, list_images, load_image

SORTER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SORTER_DIR.parent
RAW_DIR = PROJECT_DIR / "ml" / "dataset" / "raw"
REJECTED_DIR = PROJECT_DIR / "ml" / "dataset" / "rejected"
AUDIT_DIR = SORTER_DIR / "audit"
AUDIT_LOG = AUDIT_DIR / "audit_log.csv"
REJECTED_LOG = PROJECT_DIR / "ml" / "dataset" / "rejected_log.csv"

# Shared "wrong insect" descriptions, reused below.
OTHER_INSECTS = ["a moth", "a butterfly", "a caterpillar", "a beetle", "a ladybird", "a stink bug",
                 "a green leafhopper", "aphids", "a grasshopper", "a fly", "a spider", "a wasp"]


def without(*items):
    return [p for p in OTHER_INSECTS if p not in items]


PROMPTS = {
    "corn/fall_armyworm": {
        "keep": ["a caterpillar on a corn plant",
                 "a fall armyworm caterpillar with a Y mark on its head",
                 "corn leaves with ragged holes and sawdust-like droppings in the whorl",
                 "a mottled grey-brown moth"],
        "reject": without("a moth", "a caterpillar") + ["a ladybird larva", "a pupa"],
    },
    "corn/asian_corn_borer": {
        "keep": ["a caterpillar boring into a corn stalk",
                 "corn leaves with rows of small pinholes",
                 "a corn stalk with bore holes and droppings",
                 "a small yellowish-brown moth with wavy lines"],
        "reject": without("a moth", "a caterpillar") + ["a large hawk moth", "a large brown moth on a white background"],
    },
}


# Per-class cutoffs, chosen by looking at each class's contact sheet (2026-10-14 review).
# A photo is flagged when its keep score is below the cutoff.
#   None  = flag nothing (CLIP's flags were wrong for this class)
#   "all" = flag everything (nothing in the folder fits the class)
CLASS_THRESHOLDS = {
    "corn/fall_armyworm": 0.3,      # real armyworm-type caterpillars scored 0.4-0.5
    "corn/asian_corn_borer": 0.5,   # 0.1-0.5 band was random moth species
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="move flagged photos to ml/dataset/rejected/")
    p.add_argument("--threshold", type=float, default=None,
                   help="one cutoff for every class, overriding CLASS_THRESHOLDS")
    p.add_argument("--class", dest="only", help="audit one class only, e.g. corn/fall_armyworm")
    return p.parse_args()


def contact_sheet(scored, threshold, title, out_path, per_row=10, tile=130):
    """4 rows: surest rejects, borderline rejects, borderline keeps, surest keeps."""
    rejects = [s for s in scored if s[1] < threshold]
    keeps = [s for s in scored if s[1] >= threshold]
    rows = [("sure reject", rejects[:per_row]),
            ("borderline reject", rejects[-per_row:]),
            ("borderline keep", keeps[:per_row]),
            ("sure keep", keeps[-per_row:])]
    label_w, header_h = 150, 30
    sheet = Image.new("RGB", (label_w + per_row * tile, header_h + len(rows) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{title}   kept {len(keeps)}  flagged {len(rejects)}  (threshold {threshold})", fill="black")
    for r, (name, items) in enumerate(rows):
        y = header_h + r * tile
        draw.text((8, y + tile // 2 - 6), name, fill="black")
        for c, (path, keep_prob, _) in enumerate(items):
            try:
                img = load_image(path).resize((tile - 4, tile - 4))
            except Exception:
                continue
            x = label_w + c * tile
            sheet.paste(img, (x + 2, y + 2))
            draw.rectangle((x + 2, y + 2, x + 42, y + 16), fill="black")
            draw.text((x + 4, y + 3), f"{keep_prob:.2f}", fill="white")
    sheet.save(out_path, quality=90)


def unique_destination(dest: Path) -> Path:
    i = 2
    candidate = dest
    while candidate.exists():
        candidate = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        i += 1
    return candidate


def main():
    args = parse_args()
    classes = [args.only] if args.only else list(PROMPTS)
    unknown = [c for c in classes if c not in PROMPTS]
    if unknown:
        raise SystemExit(f"no prompts defined for: {unknown}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-B/32 on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    scale = model.logit_scale.exp().item()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    log_rows, moved_rows = [], []
    print(f"\n{'class':<26}{'photos':>8}{'kept':>8}{'flagged':>9}   {'cutoff'}")
    for cls in classes:
        paths = list_images(RAW_DIR / cls)
        if not paths:
            print(f"{cls:<26}{0:>8}")
            continue
        keep_p, reject_p = PROMPTS[cls]["keep"], PROMPTS[cls]["reject"]
        prompts = keep_p + reject_p
        with torch.no_grad():
            text = model.encode_text(tokenizer([f"a photo of {p}" for p in prompts]).to(device))
            text = torch.nn.functional.normalize(text.float(), dim=-1).cpu().numpy()

        emb, _ = embed_images(paths, model, preprocess, device, batch_size=64, workers=0)
        scored = []
        for path, e in zip(paths, emb):
            if e is None:
                continue
            logits = scale * (text @ e)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            keep_prob = float(probs[:len(keep_p)].sum())
            scored.append((path, keep_prob, prompts[int(probs.argmax())]))
        scored.sort(key=lambda s: s[1])

        setting = args.threshold if args.threshold is not None else CLASS_THRESHOLDS.get(cls, 0.5)
        cutoff = {None: -1.0, "all": 2.0}.get(setting, setting)
        flagged = [s for s in scored if s[1] < cutoff]
        print(f"{cls:<26}{len(scored):>8}{len(scored) - len(flagged):>8}{len(flagged):>9}   cutoff: {setting}")
        contact_sheet(scored, cutoff, cls, AUDIT_DIR / f"{cls.replace('/', '_')}.jpg")

        for path, keep_prob, best in scored:
            decision = "flagged" if keep_prob < cutoff else "kept"
            log_rows.append({"path": str(path.relative_to(RAW_DIR)), "class": cls,
                             "keep_score": f"{keep_prob:.3f}", "best_match": best, "decision": decision})
        if args.apply:
            dest_dir = REJECTED_DIR / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for path, keep_prob, best in flagged:
                dest = unique_destination(dest_dir / path.name)
                shutil.move(str(path), str(dest))
                moved_rows.append({"source": str(path.relative_to(RAW_DIR)),
                                   "destination": str(dest.relative_to(PROJECT_DIR / "ml" / "dataset")),
                                   "keep_score": f"{keep_prob:.3f}", "best_match": best})

    with open(AUDIT_LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "keep_score", "best_match", "decision"])
        w.writeheader()
        w.writerows(log_rows)
    print(f"\nLog: {AUDIT_LOG}\nContact sheets: {AUDIT_DIR}")

    if args.apply:
        write_header = not REJECTED_LOG.exists()
        with open(REJECTED_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source", "destination", "keep_score", "best_match"])
            if write_header:
                w.writeheader()
            w.writerows(moved_rows)
        print(f"Moved {len(moved_rows)} photo(s) to {REJECTED_DIR}. Log: {REJECTED_LOG}")
    else:
        print("Check only, nothing was moved. Re-run with --apply to move flagged photos.")


if __name__ == "__main__":
    main()
