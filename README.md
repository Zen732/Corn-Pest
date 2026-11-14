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

## App flow

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
8. Backend `/predict` (corn check -> severity -> detector -> pest classifier)
9. Flutter screens
10. Testing with photos from Cabadbaran corn fields

Rice and banana were removed from the project on 15 October 2026. Their photos are kept in
`ml/dataset/archive_rice_banana/` (not used). `ml/dataset/not_corn/` holds 10,000 photos of
other plants (rice, banana, tomato, cassava, cashew) that teach the corn check what is not
corn; see [docs/dataset_sources.md](docs/dataset_sources.md).
