"""Sanity checks for severity.py on synthetic photos with a known damage %. Needs no photos.

    ml/.venv/Scripts/python.exe ml/src/check_severity.py
"""
import cv2
import numpy as np

from severity import assess, grade

rng = np.random.default_rng(0)
GREEN, BROWN, YELLOW, WHITE, BLACK = (60, 140, 50), (120, 80, 40), (220, 200, 60), (245, 245, 245), (10, 10, 10)
failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    failures += not ok


def photo(background, spot_colour=None, spot_share=0.0, hole_share=0.0, size=400):
    """An elliptical green 'leaf' on a background, with round spots (damage) and holes."""
    img = np.full((size, size, 3), background, np.uint8)
    body = np.zeros((size, size), np.uint8)
    cv2.ellipse(body, (size // 2, size // 2), (170, 110), 30, 0, 360, 1, -1)
    body = body.astype(bool)
    img[body] = GREEN
    target = {"spots": spot_share, "holes": hole_share}
    painted = {"spots": np.zeros_like(body), "holes": np.zeros_like(body)}
    inner = cv2.erode(body.astype(np.uint8), np.ones((25, 25))).astype(bool)
    for kind, colour in (("spots", spot_colour), ("holes", background)):
        while painted[kind].sum() < target[kind] * body.sum():
            y, x = rng.integers(0, size, 2)
            if not body[y, x]:
                continue
            m = np.zeros((size, size), np.uint8)
            cv2.circle(m, (int(x), int(y)), int(rng.integers(6, 16)), 1, -1)
            m = m.astype(bool) & body & ~painted["spots"] & ~painted["holes"]
            if kind == "holes" and (inner & m).sum() < m.sum():  # holes stay inside the outline
                continue
            img[m] = colour
            painted[kind] |= m
    img = (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
    true_pct = 100 * (painted["spots"].sum() + painted["holes"].sum()) / body.sum()
    return img, true_pct


def near(name, img, true_pct, tolerance=4.0):
    r = assess(img)
    ok = r["damage_percent"] is not None and abs(r["damage_percent"] - true_pct) <= tolerance
    check(name, ok, f"(true {true_pct:.1f}%, measured {r['damage_percent']}% -> {r['level']})")
    return r


near("healthy leaf on white table", *photo(WHITE))
near("leaf, 30% brown spots, white table", *photo(WHITE, BROWN, 0.30))
near("leaf, 12% brown spots, black background", *photo(BLACK, BROWN, 0.12))
near("leaf, 25% yellowing", *photo(WHITE, YELLOW, 0.25))
r = near("leaf, 15% chewed holes", *photo(WHITE, hole_share=0.15))
check("holes are reported separately", r["hole_percent"] > 10, f"(hole_percent {r['hole_percent']})")
near("leaf, 10% spots + 10% holes", *photo(WHITE, BROWN, 0.10, 0.10))

check("grade bands", [grade(p) for p in (2, 15, 30, 60)] == ["Healthy", "Low", "Moderate", "High"])
blank = np.full((300, 300, 3), WHITE, np.uint8)
check("photo with no plant is 'Unclear', not a number", assess(blank)["level"] == "Unclear")
img, _ = photo(WHITE, BROWN, 0.2)
check("same photo gives the same answer twice", assess(img)["damage_percent"] == assess(img)["damage_percent"])

print("all checks passed" if not failures else f"{failures} check(s) failed")
