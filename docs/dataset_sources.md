# Dataset Sources

Where every training photo came from, under what licence, and how it was cleaned.
Per-photo credits: `ml/dataset/inaturalist_credits.csv` and `ml/dataset/dataset_credits.csv`.
Counts as of 7 November 2026. Pest list: fall armyworm, corn planthopper, corn aphid,
armyworm, common cutworm (Asian corn borer and corn weevil dropped; Asian corn borer photos
archived in `ml/dataset/archive_asian_corn_borer/`).

## Corn photos (`ml/dataset/raw/corn/`, 28,083 photos)

| Class | Photos | Sources |
|---|---|---|
| fall_armyworm | 5,000 | caterpillars/damage: Roboflow fall armyworm field photos 1,535, iNaturalist research grade 799, corn leaf pests Cameroon 641 (field photos of damaged corn), iNaturalist needs ID 25; adult moths: iNaturalist 2,000 |
| common_cutworm | 5,000 | caterpillars/damage: iNaturalist 2,773, GBIF (other sources) 85; adult moths: iNaturalist 2,142 |
| corn_aphid | 3,981 | IP102 "aphids" 3,912, iNaturalist research grade 69 |
| armyworm | 2,061 | caterpillars/damage: IP102 "army worm" 764, Kaggle Dangerous Farm Insects 56, iNaturalist 58, GBIF 1; adult moths: iNaturalist 1,101, IP102 79, Kaggle 2 |
| corn_planthopper | 2,041 | corn planthopper (*Peregrinus maidis*) 924: iNaturalist first photo of each observation 512 (needs ID 309, research grade 203), other photos of the same observations and casual observations 412 (every openly licensed photo of the species on iNaturalist); look-alike planthoppers (same family, Delphacidae) 1,117: small brown planthopper 785, white-backed planthopper 282, brown planthopper 50 (iNaturalist, credited "(look-alike)") |
| healthy | 5,000 | Doctor Maiz field photos 4,000, PlantVillage 602, first download 300, CCMT 98 |
| unidentified_damage | 5,000 | Doctor Maiz field photos 4,000 (lethal necrosis 1,657, northern leaf blight 1,520, gray leaf spot 751, common rust 72), PlantVillage 703 (rust, northern leaf blight, gray leaf spot), CCMT 297 (leaf blight, leaf spot, streak virus) |

Several photos of one iNaturalist observation (`inat_<id>.jpg`, `inat_<id>_p2.jpg`, ...) show
the same insect, so `ml/src/make_split.py` puts them all in the same set (train, val or test);
otherwise the test set would contain insects the model was trained on. See
"Train / validation / test split" below.

"First download" = photos kept from the original download (an IP102-based collection
sorted by visual similarity), after the cleaning below. "Needs ID" = iNaturalist
observations whose species was named by the observer but not yet confirmed by a second
identifier.

**Caterpillars first, then moths.** Most iNaturalist photos of fall armyworm, common cutworm
and armyworm are adult moths at night lights. Farmers photograph caterpillars and damaged
plants, so CLIP sorted every photo into caterpillar / damage / moth / other and caterpillars
and damage were collected first (`inat_larvae.py`, `filter_moths.py`). When no more caterpillar
photos could be found, adult moths were added to reach the target (`add_moths.py`; drawings
skipped): common cutworm 2,142 and armyworm 1,182. So that all three caterpillar pests have a
similar share of moths (fall armyworm 40%, common cutworm 43%, armyworm 57%), fall armyworm
swapped 2,000 of its least reliable photos (197 uncredited first-download, 1,803 unconfirmed
"needs ID") for 2,000 moths (`add_moths.py --swap 2000`); otherwise the model could learn
"adult moth = cutworm or armyworm". Still in `ml/dataset/reserve/corn/` (logged in
`moth_moves.csv`): fall armyworm 1,322 moths and 2,000 swapped-out photos, common cutworm
1,689 moths, armyworm 26 moth drawings, and 508 "other" photos (pupae, eggs, diagrams, some
caterpillars). Rejected iNaturalist observations: `dataset_sorter/inat_rejected.csv`.

## "Not corn" photos for the corn check (`ml/dataset/not_corn/`, 10,000 photos)

Other plants, used only to teach the corn check what is not a corn plant: rice 3,753, banana
1,903, tomato 1,417, cassava 1,461, cashew 1,466 (healthy and diseased of each). Filled by
`fill_not_corn.py` from the archived rice/banana classes, the archived AgML rice files, and
CCMT raw field photos of cashew, cassava and tomato (Ghana; the same collection and photo
style as the CCMT corn photos, so the corn check cannot tell corn apart by photo style alone).
Only plant photos are used, never insect close-ups: the corn check compares
corn plant photos, and insect close-ups in "not corn" would teach it that a pest close-up
means "not corn". Pest-folder photos were used only where labeled "no insect visible"
(rice hopperburn and deadhearts, banana weevil damage on pseudostems).

| Source | Photos |
|---|---|
| Project-AgML rice disease classification (Bangladesh) | 2,383 |
| Banana leaves imagery dataset, Tanzania (Mduma & Elinisa) | 998 |
| Project-AgML rice leaf disease classification (BD) | 740 |
| A Rice Leaf Disease Image Dataset for Multi-Class Disease Classification (Mendeley) | 568 |
| Banana Leaf Disease Dataset Bangladesh (Mendeley) | 341 |
| BananaLSD (Arman et al.) | 301 |
| PSFD-Musa (banana pseudostem weevil damage) | 263 |
| IP102 (rice hopperburn and deadhearts, no insect in view) | 62 |
| CCMT raw images (Ghana): cashew 1,466, cassava 1,461, tomato 1,417 | 4,344 |

## Train / validation / test split (`ml/dataset/split/`)

`ml/src/make_split.py` copies every photo into `split/train/`, `split/val/` or `split/test/`
(70 / 15 / 15). `raw/` and `not_corn/` are left as they are, and `ml/dataset/split.csv`
lists which set each photo is in.

- Each folder is split 70/15/15 on its own. Inside a corn folder, the "insect visible" yes,
  no and not-yet-labeled photos are each split 70/15/15 too.
- Photos of one iNaturalist observation (212 groups of 2 to 10 photos) stay in the same set.
  So do PlantVillage photos of the same leaf (158 leaves with 2 or 3 photos; PlantVillage's
  "copy" files are re-shots or warped copies of one leaf, grouped by its photo number, e.g.
  "R.S_HL 8315", as the dataset's authors recommend).
- The split is fixed: running the script again keeps every photo in its set and only places
  new photos.
- **train** (70%): the models learn from these. It is the only set with augmented copies
  (`aug-<kind>_<photo>`).
- **val** (15%): picks the best model (Random Forest, SVM or KNN), its settings and the pest
  classifier's confidence cutoffs.
- **test** (15%): scored once at the end and never used for any choice. These are the scores
  to report.

Real photos per folder (as of 11 Nov 2026, all labeled):

| Folder | Photos | Insect visible / damage only | train | val | test |
|---|---|---|---|---|---|
| fall_armyworm | 5,000 | 3,556 / 1,444 | 3,500 | 750 | 750 |
| common_cutworm | 5,000 | 4,998 / 2 | 3,500 | 751 | 749 |
| corn_aphid | 3,981 | 3,981 / 0 | 2,787 | 597 | 597 |
| armyworm | 2,061 | 2,023 / 38 | 1,443 | 309 | 309 |
| corn_planthopper | 2,041 | 2,041 / 0 | 1,429 | 306 | 306 |
| healthy | 5,000 | 0 / 5,000 | 3,495 | 751 | 754 |
| unidentified_damage | 5,000 | 0 / 5,000 | 3,503 | 750 | 747 |
| not_corn (10 folders) | 10,000 | - | 7,000 | 1,503 | 1,497 |
| **Total** | **38,083** | | **26,657** | **5,717** | **5,709** |

Healthy and unidentified_damage are a few photos off 3,500 / 750 / 750 because whole
PlantVillage leaves move together.

What each model uses (augmented copies only in the pest classifier's training set):

| Model | Class | train | val | test |
|---|---|---|---|---|
| Corn check | corn (every corn photo, incl. pest close-ups) | 19,657 | 4,214 | 4,212 |
| | not corn | 7,000 | 1,503 | 1,497 |
| Pest classifier (insect-visible photos + healthy) | fall_armyworm | 2,490 + 1,010 copies | 533 | 533 |
| | common_cutworm | 3,499 | 750 | 749 |
| | corn_aphid | 2,787 + 713 copies | 597 | 597 |
| | armyworm | 1,413 + 2,087 copies | 304 | 306 |
| | corn_planthopper | 1,429 + 2,071 copies | 306 | 306 |
| | healthy | 3,495 | 751 | 754 |
| Insect detector (all corn photos except healthy) | yes (insect visible) | 11,618 | 2,490 | 2,491 |
| | no (damage only) | 4,544 | 973 | 967 |
| Healthy vs damaged (corn photos with no insect in view) | healthy | 3,495 | 751 | 754 |
| | damaged (unidentified_damage + damage-only pest photos) | 4,544 | 973 | 967 |

Order to run: `make_split.py` → `extract_all.py` → `train.py` → `evaluate.py`. Rerun
`make_split.py` whenever photos are added to or removed from `raw/` or `not_corn/`.

## Sources and licences

| Source | Used for | Licence | Citation / link |
|---|---|---|---|
| iNaturalist (research grade and needs ID) | corn pests | CC0, CC BY, CC BY-NC, CC BY-SA, CC BY-NC-SA per photo | per photo in `inaturalist_credits.csv` |
| IP102 (Hugging Face copy) | corn pests | free for academic use | Wu et al., IP102, CVPR 2019 |
| PlantVillage | corn healthy and diseased | CC BY-SA 3.0 | Hughes & Salathe 2015, arXiv:1511.08060 |
| CCMT (Ghana) | corn healthy and diseased; not corn (cashew, cassava, tomato, via the Kaggle copy nirmalsankalana/crop-pest-and-disease-detection) | CC BY 4.0 | Mensah et al. 2023, doi:10.17632/bwh3zbpkpv.1 |
| Corn leaf pests, Cameroon (via Project-AgML) | fall armyworm damage | CC BY 4.0 | Tchokogoue et al. 2024, doi:10.17632/ymvghfcww7.1 |
| Roboflow Universe "fall armyworm" | fall armyworm caterpillars and damage on corn | CC BY 4.0 | https://universe.roboflow.com/fall-armyworm/fall-armyworm (2024) |
| Kaggle "Dangerous Farm Insects" | armyworm caterpillars | CC0 1.0 (as stated by the uploader) | https://www.kaggle.com/datasets/tarundalal/dangerous-insects-dataset |
| GBIF (sources other than iNaturalist, e.g. Earth Guardians, India Biodiversity Portal) | cutworm and armyworm caterpillars | CC BY / CC BY-NC per photo | per photo in `dataset_credits.csv` (GBIF occurrence link) |
| Doctor Maiz (El Salvador), Hugging Face | corn healthy and diseased (field photos) | CC BY-NC-SA 4.0 | https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies (compiled from public datasets) |
| Project-AgML rice (Hugging Face) | not corn | CC BY 4.0 | rice_disease_classification_bangladesh; rice_leaf_disease_classification_bd |
| Mendeley rice leaf disease | not corn | CC BY 4.0 | doi:10.17632/n8jpkpfz5p.1 |
| Banana leaves Tanzania | not corn | CC0 1.0 | Mduma & Elinisa, doi:10.7910/DVN/LQUWXW |
| Banana leaf disease Bangladesh | not corn | CC BY 4.0 | doi:10.17632/wfzpdmc5vx.1 |
| BananaLSD | not corn | CC BY 4.0 | doi:10.17632/9tb7k297ff.1 |
| PSFD-Musa | not corn (banana pseudostem damage) | CC BY 4.0 | Data in Brief 2022, doi:10.17632/4wyymrcpyz.1 |

CC BY-NC photos allow non-commercial use with credit: fine for a school project, not for a
commercial app. The "not corn" sources stay cited because their photos are still used.

## Cleaning

1. **First download.** The original visual-similarity sort put many wrong species into the
   pest folders. Wrong photos were moved to `ml/dataset/rejected/` after a CLIP audit with
   per-class cutoffs chosen by viewing contact sheets.
2. **Diagrams.** Drawings, labelled diagrams and infographics among the IP102 photos were
   removed (CLIP diagram score >= 0.9, checked on a contact sheet).
3. **Duplicates.** Near-identical photos (perceptual hash) are skipped within each class.
   The Roboflow fall armyworm set already contained edited copies of its photos, so
   `api_import.py` also skipped flipped/turned copies, tilted copies with black border
   wedges, and zoomed/shifted copies (CLIP similarity > 0.97). 204 that slipped through the
   first import are in `ml/dataset/rejected/corn/fall_armyworm/` (logged in `rejected_log.csv`).
   For the Doctor Maiz leaf close-ups the CLIP cutoff was 0.985, because different healthy
   leaves often score above 0.97; the closest kept pairs were checked by eye.
4. **Credits.** Every "not corn" photo has a known source; photos without one were replaced.

## Limitations (state in the paper)

- **Not on the crop.** Many insect photos (iNaturalist, IP102) show the insect on a hand,
  a wall or a white background rather than on a corn plant in the field.
- **Unconfirmed species.** About 1,800 cutworm photos (and 25 fall armyworm) are iNaturalist
  "needs ID" observations; armyworm and cutworm caterpillars look alike, so some may be a
  related species. A sample check found mostly caterpillars, with about 5-10% moths that
  CLIP mistook for caterpillars.
- **Corn aphid is several aphid species.** IP102's "aphids" class (filed under corn) shows
  aphids of several species on many plants, not only the corn leaf aphid. IP102's
  "army worm" class is likewise mixed: some photos are labelled fall armyworm or show wheat.
- **Planthoppers at lights.** Most corn planthopper photos show adults on walls or screens
  near house lights, often small in the frame, not on corn plants.
- **Small classes.** Armyworm (2,061, over half adult moths) and corn planthopper (2,041) are
  below the other pests. The pest classifier learns from insect-visible photos only, so
  fall armyworm (3,556 insect photos) and corn aphid are short too. Their training sets are
  padded to 3,500 insect photos (70% of 5,000) with augmented copies (flips, rotations,
  brighter): fall armyworm 1,010, corn aphid 713, armyworm 2,087, corn planthopper 2,071
  (`TRAIN_TARGET` in `make_split.py`). Copies are made only from training photos, so
  validation and test hold real photos only. Only the pest classifier uses them. They add
  variety but not new insects, so these classes still have fewer truly different photos
  than the others, and fewer validation and test photos (armyworm and planthopper about 300,
  fall armyworm 533, corn aphid 597, instead of 750).
- **Planthopper look-alikes.** Every photo of the corn planthopper found online is used (924).
  To reach 2,041, 1,117 photos of three rice planthoppers from the same family were added;
  they look nearly the same in photos. The class therefore means "planthopper" in general.
  This may also help with *Stenocranus pacificus*, the Mindanao corn planthopper (same
  family, no photos online). Look-alikes are marked "(look-alike)" in
  `inaturalist_credits.csv`, so they can be removed or counted separately.
  iNaturalist, IP102, Kaggle and Roboflow were all searched (November 2026): no corn
  planthopper dataset exists (the planthopper datasets are rice brown planthopper, a
  different species), and the "armyworm" datasets are mostly fall armyworm or IP102 copies.
- **Which corn planthopper?** The corn planthopper outbreaks in Mindanao (from 2003) were
  *Stenocranus pacificus*, first misidentified as *Peregrinus maidis* (PestNet 2004; UPLB
  studies). The 924 corn planthopper photos are *P. maidis*; no photos of *S. pacificus* exist on
  iNaturalist or GBIF. Photos from Cabadbaran fields are needed to cover the local species.
- **Unreachable photos.** 567 GBIF photos of cutworm caterpillars (Taiwan Moth Occurrence
  Data) could not be downloaded: the host site's security certificate has expired.
- **Healthy and damage sources.** 4,000 of each class come from one compiled dataset (Doctor
  Maiz), mostly close-up photos of single leaves. Some disease photos (especially lethal
  necrosis) show only faint streaks, so the severity measure may grade them as near healthy.
- **Corn check balance.** The corn check compares about 10,000 corn plant photos (healthy,
  damage, and pest photos with no insect in view) with 10,000 "not corn" photos. The corn
  side will grow a little once damage-only pest photos are labeled "no insect visible".
- **Adult moths in the caterpillar pests.** Adult moths make up 40% of fall armyworm, 43% of
  common cutworm and 57% of armyworm. The three species' moths are all brown noctuid moths
  and look alike, so expect more confusion between these classes on moth photos.
- **Stretched photos.** Roboflow exports every photo stretched to 640 x 640, so those fall
  armyworm photos are slightly squashed; training resizes all photos to 256 x 256 anyway.
- **Look-alike caterpillars.** Fall armyworm, common cutworm and armyworm caterpillars look
  similar; expect the pest classifier to confuse them more than other pests.
- **Lab photos.** PlantVillage photos are single leaves on a plain background, unlike
  field photos.
- **Corn check.** Its only "not corn" examples are rice and banana leaves (see
  `classification_flow.md`).
- **No field photos from Cabadbaran yet** (`ml/dataset/test_local/corn/` is empty).
