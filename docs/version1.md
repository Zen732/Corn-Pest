# Version 1 (hand-crafted features + SVM)

Version 1 is the app as it was on **22 November 2026**, before switching to the pretrained
network (Version 2). It is saved on GitHub as the branch **`version-1`** and the tag **`v1.0`**.
Read this page when you come back to Version 1, so you know how to run it and what it still
needs.

## What Version 1 is

- **Method:** every photo is described by hand-crafted features (`ml/src/features.py`: colour
  histograms, GLCM texture, LBP, HOG; 406 numbers). Four SVM models decide, and K-Means grades the
  damage (`ml/src/severity.py`).
- **Decision** (`ml/src/flow.py`, the same in the backend and in `lib/offline/flow.dart`):
  ```
  Is it a corn plant? (corn check at least 60% sure)      No  -> "doesn't look like a corn plant"
    Yes -> Is there an insect? (insect detector)
             Yes -> Which pest? (pest classifier)  50%+ sure -> pest name only
                                                   less      -> "Insect not recognized"
             No  -> Healthy, or damage Low / Moderate / High (healthy-vs-damaged model + K-Means)
  ```
- **The app:** works offline (the models run on the phone, History uploads to Supabase when
  online), login with password or a 6-digit email code, Home / History / camera / Settings.
- **App version:** 1.0.0 (`pubspec.yaml`).

### Scores (test photos never used for training)

| What | Version 1 |
|---|---|
| Whole-app test, 6,459 photos | 70.2% shown correctly |
| Pest photos named right (pest classifier alone) | 62.9% |
| When a pest name is shown, it is right | 69.5% |
| Pest photos answered "Insect not recognized" | 27.3% |
| Everyday scenes / Philippine plants rejected as "not corn" | 89.6% / 76.3% |
| Real corn photos turned away as "not corn" | 8.5% |
| 64 random phone photos rejected | 36 of 64 |

Details: [classification_flow.md](classification_flow.md). The comparison that led to Version 2
(pretrained network features: pest photos named right 87.7% instead of 62.9%) is in
`ml/results/pest_experiment/summary.txt` (`ml/src/pest_feature_experiment.py`).

## How to go back to Version 1

1. **Get the code:**
   ```
   git fetch origin
   git checkout version-1
   ```
   (`git checkout main` returns to the newest version.)
2. **Put back the Version 1 models.** The trained models are too big for GitHub, so they are
   saved on this PC in `_archive/1_old_models/version1_2026-11-22/` (checksums in
   `SHA256SUMS.txt`):
   - copy `backend_models/*` into `backend/models/`
   - copy `app_assets_models/*` into `assets/models/`

   If that folder is lost, retrain them from the dataset with the Version 1 code
   (`ml/src/make_split.py`, `extract_all.py`, `train.py`, `export_models.py`; see the
   README). Training uses a fixed random seed, so the result matches closely.
3. **Build the app:**
   `flutter build apk --release --split-per-abi --dart-define-from-file=env.json`
   (`env.json` stays on your PC; it is not on GitHub). A ready Version 1 APK was also saved as
   `CornPest_v1.apk` on the Desktop.
4. **Supabase:** Version 1 and Version 2 use the same database. Nothing to change, as long as the
   update below has been run once.

## What Version 1 still needs

Things that were planned or found but not done when work moved to Version 2:

1. **Supabase update (needed for the app to upload pest results).** Run
   `database/migrations/2026-11-22_pest_name_only.sql` once in the Supabase SQL Editor. Without
   it, pest and "Insect not recognized" scans stay on the phone as "waiting to upload".
   **Done** (run in Supabase on 22 Nov 2026; Version 2 uses the same database).
2. **Pest names are often wrong.** The pest classifier names the right pest for only 63% of test
   photos (caterpillars 57%; armyworm 37%). Tried and not better: stronger SVM settings, SIFT
   visual words, moths and caterpillars as separate classes, cropping to the insect
   (57.5%). Version 2's pretrained features were the fix that worked.
3. **Unknown insects get a pest name.** An insect that is not one of the five pests (a beetle on
   a corn leaf was "Corn aphid, 87%") is named as one of them when the classifier is sure. The
   50% rule only catches unsure answers. Planned fix: an "other insect" class trained on photos of
   other Philippine insects (iNaturalist), not done in Version 1.
4. **The corn check still lets things through:** about 1 in 4 garden plants, 1 in 10 everyday
   scenes, and pale objects on dark backgrounds (a pale bowl on a dark table). No "not corn"
   photos of floors, ground or footwear were collected.
5. **Step 10 is not done: no photos from Cabadbaran corn fields.** The folders
   `ml/dataset/test_local/corn/<pest or healthy>/` are empty. The local test (how many local corn
   photos are turned away, how many pests are named right) has never been run.
6. **"Was this right?" button** on the result screen (farmer marks the answer right or wrong and
   picks the real one, to collect labeled local photos): suggested, not built.
7. **Before publishing the app** (not needed for a school demo):
   - the app id is still `com.example.pest_assessment` (`android/app/build.gradle.kts`);
   - release builds are signed with the debug key (make a release key and keep it safe);
   - `android/app/src/main/res/xml/network_security_config.xml` still allows plain http to
     192.168.137.1 and 127.0.0.1 (for the local backend); remove it if the backend is not used.
8. **The FastAPI backend is not online.** The app does not need it (it checks photos offline);
   it is kept for testing (`backend/`).
9. **Backups:** the trained models, the photos (`ml/dataset/`), the downloaded lists and
   `env.json` are only on this PC. Copy them to Google Drive or a USB drive.

## Where things are

| What | Where |
|---|---|
| Decision rules | `ml/src/flow.py` (Python), `backend/app/services/flow.py` (exact copy), `lib/offline/flow.dart` (phone) |
| Photo features | `ml/src/features.py`, `lib/offline/features.dart` |
| Training and tests | `ml/src/train.py`, `evaluate_flow.py`, `parity_reference.py`, `test/offline_parity_test.dart` |
| Datasets and credits | `ml/dataset/`, [dataset_sources.md](dataset_sources.md) |
| Supabase setup | `database/README.md`, `database/schema.sql`, `database/migrations/` |
| Phone offline checking | [offline_checking.md](offline_checking.md) |
