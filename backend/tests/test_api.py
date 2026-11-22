"""Backend tests. Run from the backend folder:
    .venv/Scripts/python.exe -m pytest -v

The tests with real photos use ml/dataset/split/test/ and ml/results/flow_test.csv, which stay on
the training PC (not on GitHub); elsewhere they are skipped.
"""
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
FLOW_TEST = ROOT / "ml" / "results" / "flow_test.csv"
TEST_PHOTOS = ROOT / "ml" / "dataset" / "split" / "test"
OUTCOMES = {"healthy", "severity_only", "pest", "unknown_pest", "not_corn", "unclear"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the startup that loads the models
        yield c


def send(client, data, name="photo.jpg"):
    return client.post("/predict", files={"photo": (name, data, "application/octet-stream")})


def encode(rgb, ext=".png"):
    return cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


def test_shared_code_matches_ml_src():
    sys.path.insert(0, str(BACKEND))
    from sync_shared import differences
    assert differences() == [], "run backend/sync_shared.py: the backend copies differ from ml/src"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == ["corn_check", "detector", "health", "pest"]
    assert body["model_version"] not in (None, "", "unknown")


@pytest.mark.parametrize("data, status", [
    (b"", 400),                                          # empty upload
    (b"this is a text file, not a photo", 415),          # not JPG/PNG
    (b"\xff\xd8\xff" + b"garbage" * 100, 400),           # starts like a JPG but is broken
    (b"\xff\xd8\xff" + b"\0" * (10 * 1024 * 1024), 413), # over 10 MB
])
def test_bad_uploads_get_clear_errors(client, data, status):
    r = send(client, data)
    assert r.status_code == status
    assert r.json()["detail"]  # a message the app can show


@pytest.mark.parametrize("colour, size", [((255, 255, 255), (300, 300)), ((60, 140, 50), (64, 64))])
def test_plain_images_still_get_an_answer(client, colour, size):
    rgb = np.full((*size, 3), colour, np.uint8)
    r = send(client, encode(rgb), "plain.png")
    assert r.status_code == 200
    assert r.json()["outcome"] in OUTCOMES


def _sample(per_group=8, seed=0):
    if not FLOW_TEST.exists() or not TEST_PHOTOS.exists():
        pytest.skip("test photos and ml/results/flow_test.csv are only on the training PC")
    with open(FLOW_TEST, newline="", encoding="utf-8") as f:
        by_group = defaultdict(list)
        for row in csv.DictReader(f):
            by_group[row["group"]].append(row)
    rng = random.Random(seed)
    return [row for g in sorted(by_group) for row in rng.sample(by_group[g], min(per_group, len(by_group[g])))]


def _num(value):
    return None if value in ("", None) else round(float(value), 3)


def test_same_answers_as_the_whole_app_test(client):
    """About 64 test photos (8 of each kind) give exactly the answers recorded by
    ml/src/evaluate_flow.py, so the app behaves as tested. Also reports the speed."""
    rows, seconds, mismatches = _sample(), [], []
    for row in rows:
        data = (TEST_PHOTOS / row["path"]).read_bytes()
        start = time.perf_counter()
        r = send(client, data, Path(row["path"]).name)
        seconds.append(time.perf_counter() - start)
        assert r.status_code == 200, r.text
        got = r.json()
        for key in ("outcome", "pest", "severity"):
            if (got[key] or "") != row[key]:
                mismatches.append((row["path"], key, row[key], got[key]))
        for key in ("damage_percent", "corn_confidence", "pest_confidence", "insect_confidence", "health_confidence"):
            if _num(got[key]) != _num(row[key]):
                mismatches.append((row["path"], key, row[key], got[key]))
    print(f"\n{len(rows)} photos: {np.mean(seconds):.2f} s per photo on average, slowest {max(seconds):.2f} s")
    assert not mismatches, mismatches[:5]
    assert np.mean(seconds) < 5
