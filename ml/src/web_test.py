"""Version 2 check on photos the models never saw: 30 kinds of crop pests from iNaturalist
(dataset_sorter/fill_web_test.py), the 64 random phone photos (ml/dataset/test_random/) and the
user's phone screenshots (PEST_SCREENSHOTS folder: 6.png bowl, 7.jpg beetle, 8.jpg caterpillar,
9.jpg aphid; the photo part of each screen), for the current
models and, for comparison, the archived ones (Version 2 with MobileNetV3, Version 1).

Two scores per pest kind:
    pest step   the pest classifier alone, as if the corn check and detector said "corn, insect".
                One of the five pests: right if it names that pest at least 50% sure.
                Any other pest: right if it does NOT name one of the five (it says other_insect,
                or is under 50% sure) -> "Insect not recognized".
    whole app   the full decision (flow.py). One of the five: right if outcome "pest" with the
                right name. Other pests: right if the outcome is not "pest" (so "not corn" counts
                as right: many web photos are not on corn).

Run from the project root:
    ml/.venv/Scripts/python.exe ml/src/web_test.py
Writes ml/results/web_test/summary.txt and photos.csv.
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from features import decode_image
from flow import PEST_AT_LEAST, assess_photo, load_models, photo_numbers
from predict import predict_one

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "ml" / "dataset" / "test_web"
CREDITS = ROOT / "ml" / "dataset" / "test_web_credits.csv"
RANDOM_DIR = ROOT / "ml" / "dataset" / "test_random"
SCREENSHOTS = Path(os.environ.get("PEST_SCREENSHOTS", "."))
OUT = ROOT / "ml" / "results" / "web_test"
# V2 = the current models (EfficientNet-B2 since 25 Nov 2026); V2mob = Version 2 with MobileNetV3
# (23 Nov 2026, archived); V1 = hand-made features. All are checked with today's rules (flow.py).
VERSIONS = {"V2": ROOT / "backend" / "models",
            "V2mob": ROOT / "_archive" / "1_old_models" / "version2_mobilenet_2026-11-24" / "backend_models",
            "V1": ROOT / "_archive" / "1_old_models" / "version1_2026-11-22" / "backend_models"}
FIVE = {"fall_armyworm", "common_cutworm", "armyworm", "corn_aphid", "corn_planthopper"}


def read(path):
    return decode_image(np.fromfile(str(path), dtype=np.uint8).tobytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CREDITS, newline="", encoding="utf-8") as f:
        photos = [r for r in csv.DictReader(f)]
    versions = {v: load_models(folder) for v, folder in VERSIONS.items() if folder.exists()}
    rows = []
    for i, r in enumerate(photos):
        rgb = read(ROOT / "ml" / "dataset" / r["file"])
        row = {"file": r["file"], "kind": r["kind"], "expected": r["expected"]}
        for v, models in versions.items():
            numbers = {}  # filled by photo_numbers, reused by assess_photo
            x = photo_numbers(rgb, models, numbers)
            pest, conf, _ = predict_one(models["pest"], x(models["pest"]))
            named = pest if (pest in FIVE and conf >= PEST_AT_LEAST) else None
            flow = assess_photo(rgb, models, numbers)
            row.update({f"{v}_pest": pest, f"{v}_conf": round(conf, 3), f"{v}_named": named or "",
                        f"{v}_outcome": flow["outcome"], f"{v}_app_pest": flow["pest"] or "",
                        f"{v}_step_right": (named == r["expected"]) if r["expected"] in FIVE else named is None,
                        f"{v}_app_right": (flow["outcome"] == "pest" and flow["pest"] == r["expected"])
                        if r["expected"] in FIVE else flow["outcome"] != "pest"})
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(photos)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "photos.csv", index=False)

    lines = ["WEB TEST: 30 kinds of crop pests from iNaturalist, never used for training", ""]
    vs = list(versions)
    head = f"{'kind':<26}{'n':>3}  " + "  ".join(f"{v} pest step / whole app" for v in vs)
    for title, mask in (("The app's five corn pests (should be named)", df["expected"] != "other"),
                        ("Other crop pests (should NOT be named as one of the five)", df["expected"] == "other")):
        lines += [title, head]
        sub = df[mask]
        for kind, g in sub.groupby("kind", sort=False):
            cells = "  ".join(f"{100 * g[f'{v}_step_right'].mean():>5.0f}% / {100 * g[f'{v}_app_right'].mean():>5.0f}%"
                              + " " * 9 for v in vs)
            lines.append(f"{kind:<26}{len(g):>3}  {cells}")
        tot = "  ".join(f"{100 * sub[f'{v}_step_right'].mean():>5.1f}% / {100 * sub[f'{v}_app_right'].mean():>5.1f}%"
                        + " " * 7 for v in vs)
        lines += [f"{'ALL':<26}{len(sub):>3}  {tot}", ""]
    for v in vs:
        both = df[f"{v}_step_right"].mean()
        lines.append(f"{v}: pest step right on {100 * both:.1f}% of {len(df)} photos; whole app right on "
                     f"{100 * df[f'{v}_app_right'].mean():.1f}%")
        wrong = df[(df["expected"] == "other") & (df[f"{v}_named"] != "")]
        if len(wrong):
            lines.append(f"    other pests named as one of the five: "
                         + ", ".join(f"{k} {n}" for k, n in wrong[f"{v}_named"].value_counts().items()))
        oc = df[f"{v}_outcome"].value_counts()
        lines.append("    whole-app outcomes: " + ", ".join(f"{k} {n}" for k, n in oc.items()))
    lines.append("")

    # The 64 random phone photos and the screenshots.
    extra = {}
    if RANDOM_DIR.exists():
        for p in sorted(RANDOM_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                extra[f"random/{p.name}"] = read(p)
    for name, label in (("6.png", "screenshot: bowl"), ("7.jpg", "screenshot: beetle"),
                        ("8.jpg", "screenshot: curled caterpillar (looks like a cutworm)"),
                        ("9.jpg", "screenshot: blue-green aphid (looks like a corn aphid)")):
        if (SCREENSHOTS / name).exists():
            extra[label] = cv2.cvtColor(cv2.imread(str(SCREENSHOTS / name)), cv2.COLOR_BGR2RGB)[200:610, 48:672]
    for v, models in versions.items():
        res = {k: assess_photo(rgb, models) for k, rgb in extra.items()}
        rnd = [r for k, r in res.items() if k.startswith("random/")]
        if rnd:
            lines.append(f"{v}: your {len(rnd)} random photos: "
                         + ", ".join(f"{k} {n}" for k, n in pd.Series([r["outcome"] for r in rnd]).value_counts().items()))
        for k, r in res.items():
            if k.startswith("screenshot"):
                # corn_confidence: how sure the corn check is of its own answer (corn or not corn)
                lines.append(f"{v}: {k} -> {r['outcome']}" + (f" {r['pest']}" if r["pest"] else "")
                             + f" (corn check {r['corn_confidence']:.0%} sure"
                             + (f", pest {r['pest_confidence']:.0%})" if r["pest_confidence"] is not None else ")"))
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
