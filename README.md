# Corn Pest Assessment (Cabadbaran)

Mobile app that checks a photo of a corn plant and reports one of three things: healthy,
damage severity only, or pest identification + severity. Pests: fall armyworm, corn
planthopper, corn aphid, armyworm and common cutworm. No treatment
recommendations are shown anywhere in the app.
See [docs/classification_flow.md](docs/classification_flow.md) for the full decision logic.

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
      -> Corn check: is this a corn plant?
           - No / unsure  -> Message ("doesn't look like a corn plant"), stop
           - Yes          -> continue
      -> Result:
           - Pest name + Severity                  (insect visible in photo)
           - Healthy                              (no insect, healthy-vs-damaged model says healthy)
           - Severity only (Low/Moderate/High)     (no insect, damaged; K-Means damage %)
      -> Saved to History
```

No field questions and no recommendations. Severity is computed from the photo alone.

## Build order

1. Project skeleton
2. Dataset collection and checking (target 5,000 photos per pest)
3. Feature extraction (`ml/src/features.py`, shared with backend)
4. Label each photo as insect-visible or damage-only (`ml/dataset/labels.csv`)
5. Train the corn check, insect detector and pest classifier; compare models
6. Severity (K-Means damage %), healthy/damage cutoff
7. Database schema and seed data (scans + results, no recommendations)
8. Backend `/predict` (corn check -> insect detector -> pest classifier or healthy-vs-damaged -> severity); see [backend/README.md](backend/README.md)
9. Flutter screens
10. Testing with photos from Cabadbaran corn fields

Rice and banana were removed from the project on 15 October 2026. Their photos are kept in
`_archive/2_unused_photos/rice_banana_project/` (not used). `ml/dataset/not_corn/` holds 15,000 photos that
teach the corn check what is not corn: 10,000 of other crops (rice, banana, tomato, cassava, cashew),
2,500 plants photographed in the Philippines and 2,500 everyday scenes; see
[docs/dataset_sources.md](docs/dataset_sources.md). Whole-app test: 74.5% of 6,459 test photos
shown correctly (see [docs/classification_flow.md](docs/classification_flow.md)).
