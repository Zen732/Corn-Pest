"""Report on train.py's saved metrics: scores, the corn check's background test, and a
confusion matrix image per model.

Run after train.py, from the project root:
    ml/.venv/Scripts/python.exe ml/src/evaluate.py

Writes ml/results/<model>_confusion.png for each model.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METRICS_DIR = RESULTS_DIR / "metrics"


def plot_confusion(cm, classes, title, out_path):
    cm = np.array(cm)
    size = max(4, 1.1 * len(classes))
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    files = sorted(METRICS_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No metrics found. Run train.py first.")

    print("Scores are on the test set (never used for any choice); val f1 is what picked the model.")
    print(f"{'model':<18}{'best':<15}{'val f1':>8}{'acc':>7}{'f1':>7}{'train real':>12}{'+copies':>9}"
          f"{'val':>6}{'test':>6}")
    for f in files:
        m = json.load(open(f, encoding="utf-8"))
        name = f.stem
        val_f1 = m.get("val", {}).get("f1_macro")
        print(f"{name:<18}{m['best_model']:<15}{val_f1 if val_f1 is not None else float('nan'):>8.3f}"
              f"{m['test_accuracy']:>7.3f}{m['test_f1_macro']:>7.3f}"
              f"{m.get('n_train_original', m.get('n_train', 0)):>12}{m.get('n_train_augmented', 0):>9}"
              f"{m.get('n_val', 0):>6}{m['n_test']:>6}")
        if m.get("cutoffs"):
            plain = m["without_cutoffs"]["test_f1_macro"]
            print(f"    cutoffs {m['cutoffs']}: f1 {plain:.3f} without them -> {m['test_f1_macro']:.3f}")
        for cls, rep in m["classification_report"].items():
            if cls in m["classes"] and rep["f1-score"] < 0.70:
                print(f"    weak class '{cls}': f1 {rep['f1-score']:.2f} (precision {rep['precision']:.2f}, "
                      f"recall {rep['recall']:.2f}), {int(rep['support'])} test photos")
        plot_confusion(m["confusion_matrix"], m["classes"], f"{name} ({m['best_model']})",
                       RESULTS_DIR / f"{name}_confusion.png")

    check_file = METRICS_DIR / "corn_check.json"
    check = json.load(open(check_file, encoding="utf-8")) if check_file.exists() else {}
    bg = check.get("background_check")
    if bg:
        print("\nCorn check background test:")
        print(f"  telling corn from not-corn by the photo's border colour alone: {bg['border_colour_only_accuracy']:.3f}"
              f" (chance {bg['chance_accuracy']:.3f}, real model {check['test_accuracy']:.3f})")
        for kind, r in bg["accuracy_by_background"].items():
            print(f"  real model on {kind:<8} backgrounds: {r['accuracy']:.3f} ({r['n']} test photos)")
        print("  background share, corn vs not corn:")
        for c, shares in bg["background_share_per_crop"].items():
            print(f"    {c:<8} " + "  ".join(f"{k} {v:.0%}" for k, v in shares.items()))

    print(f"\nConfusion matrix images saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
