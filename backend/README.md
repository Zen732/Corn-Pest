# Backend (FastAPI)

A small web server the app sends each photo to. It runs the four trained models with the same
decision as the whole-app test (`ml/src/flow.py`) and sends back the result. It never touches
the database; the app saves results to Supabase itself (see [database/README.md](../database/README.md)).

| Address | What it does |
|---|---|
| `GET /health` | Is the server running? Lists the loaded models and the model version |
| `POST /predict` | Send one photo (form field `photo`, JPG or PNG, at most 10 MB) and get the result |
| `GET /docs` | A page in the browser where you can try both, no app needed |

Example answer from `/predict`:

```json
{
  "outcome": "pest",
  "pest": "fall_armyworm",
  "pest_confidence": 0.84,
  "severity": "High",
  "damage_percent": 100.0,
  "insect_visible": true,
  "insect_confidence": 0.973,
  "healthy": null,
  "health_confidence": null,
  "corn_confidence": 0.984,
  "model_version": "2026-11-14"
}
```

**`outcome`** is one of:

| `outcome` | Meaning | Save to Supabase? |
|---|---|---|
| `healthy` | Healthy corn | Yes |
| `severity_only` | Damage, no insect in the photo. Show `severity` and `damage_percent` | Yes |
| `pest` | Show `pest` and `severity` | Yes |
| `not_corn` | Show "This doesn't look like a corn plant" | No |
| `unclear` | Ask for a clearer photo | No |

**Errors** come back with a `detail` message the app can show:

| Code | Meaning |
|---|---|
| 400 | Empty or damaged photo |
| 413 | The photo is over 10 MB |
| 415 | Not a JPG or PNG |

## First-time setup

The `.venv` folder may already exist on the training PC. Otherwise, from the project root:

```
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

The package versions in `requirements.txt` must match the ones the models were trained with
(`ml/.venv`); the model files don't load with other scikit-learn versions.

The model files (`backend/models/*.joblib`, about 100 MB) aren't on GitHub. They're made by
`ml/src/train.py`; copy them over if you set this up on another computer.

## Running it

From the `backend` folder:

```
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **On this PC:** open http://localhost:8000/docs in a browser to try a photo.
- **From your phone:** the phone and the PC must be on the same Wi-Fi.
  1. Run `ipconfig` and find the PC's IPv4 address, for example `192.168.1.5`.
  2. The app then uses `http://192.168.1.5:8000`.
  3. If Windows asks whether to allow Python on the network, allow it for **private** networks.
- **Stop it:** press Ctrl+C.

Anyone who knows the address can use the server; there is no login check. That's fine on home
Wi-Fi. Putting it online so it works anywhere is a later step.

## Tests

From the `backend` folder:

```
.venv/Scripts/python.exe -m pytest -v -s
```

The tests check four things:

- **Shared code:** the backend's copies of the shared code match `ml/src`.
- **Health check:** `/health` lists all four models.
- **Bad uploads:** empty, text, broken and too-big files get clear errors.
- **Same answers as the whole-app test:** 64 real test photos (8 of each kind) give exactly the
  answers recorded by `ml/src/evaluate_flow.py`. This also prints the speed: about 0.15 s per
  photo on the training PC.

The real-photo test needs the test photos and `ml/results/flow_test.csv`, which stay on the
training PC; elsewhere it's skipped.

## After retraining or changing the shared code

`app/services/` holds exact copies of `ml/src/features.py`, `severity.py`, `predict.py` and
`flow.py`. Never edit them here. After changing them in `ml/src`, or after retraining:

```
backend/.venv/Scripts/python.exe backend/sync_shared.py
```

Then restart the server. `train.py` updates `models/model_version.txt` to the training date, so
saved scans show which models made them.
