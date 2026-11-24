# Corn Pest Assessment (Cabadbaran)

Mobile app that checks a photo of a corn plant and reports one of three things: healthy,
damage severity only, or pest identification + severity. Pests: fall armyworm, corn
planthopper, corn aphid, armyworm and common cutworm. No treatment
recommendations are shown anywhere in the app.
See [docs/classification_flow.md](docs/classification_flow.md) for the full decision logic.

**Versions:** this is **Version 2** (23 Nov 2026; app 2.1.0 since 25 Nov): the photo is turned
into 1408 numbers by a ready-made image network (EfficientNet-B2, run on the phone; MobileNetV3-Large
on 23-24 Nov) and the same kind of SVMs decide; an "other insect" class catches insects that are
not one of the five pests. See [docs/version2.md](docs/version2.md) for what changed, the scores
and how to retrain.
Version 1 (hand-crafted features + SVM) is saved as the branch `version-1` / tag `v1.0`;
see [docs/version1.md](docs/version1.md) for how to go back to it and what it still needs.

## Layout

| Folder | Purpose |
|--------|---------|
| `lib/`, `android/`, `ios/`, ... | Flutter mobile app |
| `ml/` | Dataset, training code (feature extraction, corn check, insect detector, pest classifier, healthy-vs-damaged model, severity, whole-app test) |
| `dataset_sorter/` | Tools that collected and checked the training photos |
| `backend/` | FastAPI server exposing `POST /predict` |
| `database/` | Supabase `schema.sql` and `seed_data.sql` (scans and results only, no recommendations) |
| `docs/` | Design, severity method, dataset sources, corn pest research |
| `_archive/` | Unused photos, downloads, old models, backups and logs, sorted by type (local only, not on GitHub; see `_archive/README.md`) |

## App flow

The app opens on the login form (email + password). Under it: **Sign in or sign up with an
email code** (a 6-digit code by email, no password; new emails get an account automatically,
then the app asks the name and barangay once) and **Create account** (password account, confirmed
with the same kind of emailed code). Every email has a code, never a link.

Signed in, a bottom bar holds **Home** (greeting, numbers, recent scans, photo tip), **History**,
a big round **camera button** (Scan) and **Settings** (your info, edit name and barangay, change
password, upload waiting scans, sign out). Scans work offline: the models run on the phone and
History uploads to Supabase when there is internet (see [docs/offline_checking.md](docs/offline_checking.md)).

```
Login -> Photo tips -> Take/pick photo
      -> Corn check: is this a corn plant? (at least 50% sure; 60% in Version 1)
           - No / unsure  -> Message ("doesn't look like a corn plant"), stop
           - Yes          -> continue
      -> Is there an insect?
           - Yes -> Which pest?
                      - at least 50% sure  -> Pest name (no severity)
                      - less sure, or "other insect" -> "Insect not recognized"
           - No  -> Healthy, or Severity only (Low/Moderate/High; K-Means damage %)
      -> Saved to History
```

No field questions and no recommendations. Severity is computed from the photo alone.

## Build order

1. Project skeleton
2. Dataset collection and checking (target 5,000 photos per pest)
3. Feature extraction (`ml/src/features.py`, shared with backend; Version 2: the image network,
   `ml/src/cnn_features.py`, `network_export.py`, `network.py`, `lib/offline/network.dart`)
4. Label each photo as insect-visible or damage-only (`ml/dataset/labels.csv`)
5. Train the corn check, insect detector and pest classifier; compare models
6. Severity (K-Means damage %), healthy/damage cutoff
7. Database schema and seed data (scans + results, no recommendations)
8. Backend `/predict` (corn check -> insect detector -> pest classifier or healthy-vs-damaged -> severity); see [backend/README.md](backend/README.md)
9. Flutter screens
10. Testing with photos from Cabadbaran corn fields

Rice and banana were removed from the project on 15 October 2026. Their photos are kept in
`_archive/2_unused_photos/rice_banana_project/` (not used). `ml/dataset/not_corn/` holds 16,600 photos that
teach the corn check what is not corn: 10,000 of other crops (rice, banana, tomato, cassava, cashew),
2,500 plants photographed in the Philippines, 2,500 everyday scenes and (Version 2) 1,600 close-ups of
small objects (rice grains, coffee beans, pebbles, shells); see
[docs/dataset_sources.md](docs/dataset_sources.md). `ml/dataset/raw/corn/other_insect/` holds
5,026 photos of insects and spiders that are not one of the five pests. Whole-app test
(Version 2, EfficientNet-B2): 93.1% of 7,453 test photos shown correctly (Version 1: 70.2% of
6,459; Version 2 on those same 6,459: 93.2%); a pest name, when shown, is right 91.9% of the
time (Version 1: 69.5%). On 300 web photos of 30 kinds of pests the app never saw: 76.7% right
(Version 1: 51.7%; MobileNetV3: 74.7%). See [docs/version2.md](docs/version2.md) and
[docs/classification_flow.md](docs/classification_flow.md).
