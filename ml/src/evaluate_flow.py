"""Whole-app test: run every test photo through the app's full decision (flow.py: corn check ->
insect detector -> severity -> pest classifier) and compare what the app would show with what
it should show. Test photos were never used for any choice during training.

Run after train.py, from the project root:
    ml/.venv/Scripts/python.exe ml/src/evaluate_flow.py

What each test photo should give:
    other plants (not_corn/)                     -> not_corn
    healthy/                                     -> healthy
    unidentified_damage/, and pest-folder photos
    labeled "damage only" (no insect in view)    -> severity_only
    pest-folder photos with the insect in view   -> pest, with the right pest name

Writes ml/results/flow_test.csv (one row per photo) and ml/results/flow_summary.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features import decode_image
from flow import assess_photo, load_models
from make_split import SPLIT_DIR

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "ml" / "results"
MODELS_DIR = ROOT / "backend" / "models"
_models = {}


def _run(path, features):
    """One photo, in a worker process (models are loaded once per worker)."""
    if not _models:
        _models.update(load_models(MODELS_DIR))
    rgb = decode_image(np.fromfile(str(path), dtype=np.uint8).tobytes())
    return assess_photo(rgb, _models, features)


def expected(row):
    if row["crop"] == "not_corn":
        return "not_corn", "other plant"
    if row["pest_class"] == "healthy":
        return "healthy", "healthy corn"
    if row["pest_class"] == "unidentified_damage" or row["bug_visible"] == "no":
        return "severity_only", "damage, no insect"
    return "pest", f"{row['pest_class']} (insect)"


def main():
    X = np.load(RESULTS_DIR / "features.npy")
    meta = pd.read_csv(RESULTS_DIR / "features_meta.csv", keep_default_na=False)
    test = meta[(meta["split"] == "test") & (meta["aug"] == "")].copy()
    print(f"{len(test)} test photos", flush=True)
    results = Parallel(n_jobs=-1, verbose=2)(
        delayed(_run)(SPLIT_DIR / "test" / p, X[i]) for i, p in zip(test.index, test["path"]))

    out = pd.concat([test[["path", "crop", "pest_class", "bug_visible"]].reset_index(drop=True),
                     pd.DataFrame(results)], axis=1)
    out[["expected", "group"]] = out.apply(lambda r: pd.Series(expected(r)), axis=1)
    out["shown"] = np.where(out["outcome"] == "pest", out["pest"], out["outcome"])
    out["correct"] = (out["outcome"] == out["expected"]) & (
        (out["expected"] != "pest") | (out["pest"] == out["pest_class"]))
    out.to_csv(RESULTS_DIR / "flow_test.csv", index=False)

    table = pd.crosstab(out["group"], out["shown"], normalize="index").mul(100).round(1)
    summary = out.groupby("group").agg(photos=("correct", "size"), correct_pct=("correct", "mean"))
    summary["correct_pct"] = (summary["correct_pct"] * 100).round(1)
    summary.loc["ALL"] = [len(out), round(100 * out["correct"].mean(), 1)]
    summary.to_csv(RESULTS_DIR / "flow_summary.csv")
    print("\nWhat the app would show (% of each kind of test photo):")
    print(table.to_string())
    print("\nShown correctly:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
