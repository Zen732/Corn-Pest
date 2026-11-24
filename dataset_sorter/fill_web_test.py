"""Version 2 web test: photos of 30 kinds of crop pests from iNaturalist, to check the app on
insects it has never seen. Never used for training (ml/dataset/test_web/ is outside raw/).

    5 kinds   the app's own corn pests: the app should name them
    25 kinds  other crop pests (corn borers, rice bugs, fruit flies, beetles ...): the app should
              NOT name them as one of its five pests ("Insect not recognized" is right)

Per kind up to PER_KIND research-grade photos with open licences, newest first. Skipped: any
observation already in the dataset (inaturalist_credits.csv) and near-duplicates of the photos
in the matching training folder. Credited in ml/dataset/test_web_credits.csv.
iNaturalist is asked about once per second with the project's generic User-Agent.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/fill_web_test.py
"""
import csv
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from fill_real_world import known_hashes
from inat_download import (API, CREDITS_FILE, DATASET_DIR, LICENSES, RAW_DIR, append_csv, fetch_photo, first_open_photo,
                           get_json, is_duplicate, list_images)

OUT = DATASET_DIR / "test_web"
CREDITS = DATASET_DIR / "test_web_credits.csv"
FIELDS = ["file", "kind", "expected", "observation_id", "observation_url", "species", "license", "attribution",
          "observed_on", "place"]
PER_KIND = 10
# kind (folder), iNaturalist name, what the app should say: one of its five pests, or "other"
KINDS = [
    ("fall_armyworm", "Spodoptera frugiperda", "fall_armyworm"),
    ("common_cutworm", "Spodoptera litura", "common_cutworm"),
    ("armyworm", "Mythimna separata", "armyworm"),
    ("corn_aphid", "Rhopalosiphum maidis", "corn_aphid"),
    ("corn_planthopper", "Peregrinus maidis", "corn_planthopper"),
    ("asian_corn_borer", "Ostrinia furnacalis", "other"),
    ("cotton_bollworm", "Helicoverpa armigera", "other"),
    ("black_cutworm", "Agrotis ipsilon", "other"),
    ("green_stink_bug", "Nezara viridula", "other"),
    ("rice_bug", "Leptocorisa oratoria", "other"),
    ("silverleaf_whitefly", "Bemisia tabaci", "other"),
    ("diamondback_moth", "Plutella xylostella", "other"),
    ("eggplant_fruit_borer", "Leucinodes orbonalis", "other"),
    ("oriental_fruit_fly", "Bactrocera dorsalis", "other"),
    ("melon_fly", "Zeugodacus cucurbitae", "other"),
    ("migratory_locust", "Locusta migratoria", "other"),
    ("javanese_grasshopper", "Valanga nigricornis", "other"),
    ("yellow_stem_borer", "Scirpophaga incertulas", "other"),
    ("rice_leaffolder", "Cnaphalocrocis medinalis", "other"),
    ("rhinoceros_beetle", "Oryctes rhinoceros", "other"),
    ("red_palm_weevil", "Rhynchophorus ferrugineus", "other"),
    ("pumpkin_beetle", "Aulacophora", "other"),
    ("28_spotted_ladybird", "Henosepilachna vigintioctopunctata", "other"),
    ("red_cotton_stainer", "Dysdercus cingulatus", "other"),
    ("cabbage_white", "Pieris rapae", "other"),
    ("lime_swallowtail", "Papilio demoleus", "other"),
    ("two_spotted_spider_mite", "Tetranychus urticae", "other"),
    ("papaya_mealybug", "Paracoccus marginatus", "other"),
    ("banana_weevil", "Cosmopolites sordidus", "other"),
    ("banana_skipper", "Erionota thrax", "other"),
]


def used_ids():
    ids = set()
    for f in (CREDITS_FILE, CREDITS):
        if f.exists():
            with open(f, newline="", encoding="utf-8") as fh:
                ids |= {r["observation_id"] for r in csv.DictReader(fh)}
    return ids


def candidates(name, used, want=3 * PER_KIND, pages=15):
    """Observations not yet used, newest first, paging back through older ones. Research grade
    first; only if those run out, "needs ID" ones too (marked in the credits)."""
    out = []
    for grade in ("research", "needs_id"):
        id_below = None
        for _ in range(pages):
            q = {"taxon_name": name, "quality_grade": grade, "photos": "true",
                 "photo_license": ",".join(sorted(LICENSES)), "order_by": "id", "order": "desc", "per_page": 200,
                 **({"id_below": id_below} if id_below else {})}
            obs = get_json(f"{API}?{urllib.parse.urlencode(q)}")["results"]
            time.sleep(1)
            if not obs:
                break
            for o in obs:
                p = first_open_photo(o)
                if p and str(o["id"]) not in used:
                    o["grade"] = grade
                    out.append((o, p))
            id_below = obs[-1]["id"]
            if len(out) >= want:
                return out[:want]
    return out


def main():
    used = used_ids()
    with ThreadPoolExecutor(4) as pool:
        for kind, name, expected in KINDS:
            folder = OUT / kind
            folder.mkdir(parents=True, exist_ok=True)
            have = len(list_images(folder))
            if have >= PER_KIND:
                print(f"{kind}: {have} already")
                continue
            train_folder = RAW_DIR / "corn" / (expected if expected != "other" else "other_insect")
            known = known_hashes([train_folder, folder])
            picks = candidates(name, used)
            got = pool.map(fetch_photo, [p["url"].replace("/square.", "/medium.") for _, p in picks])
            added = 0
            for (o, p), res in zip(picks, got):
                if res is None or have + added >= PER_KIND:
                    continue
                img, h = res
                if is_duplicate(h, known):
                    continue
                known.append(h)
                used.add(str(o["id"]))
                fname = f"inat_{o['id']}.jpg"
                img.save(folder / fname, quality=92)
                append_csv(CREDITS, FIELDS, {
                    "file": f"test_web/{kind}/{fname}", "kind": kind, "expected": expected,
                    "observation_id": o["id"], "observation_url": f"https://www.inaturalist.org/observations/{o['id']}",
                    "species": (o.get("taxon") or {}).get("name", name) + ("" if o["grade"] == "research" else " (needs_id)"),
                    "license": p["license_code"],
                    "attribution": p.get("attribution", ""), "observed_on": o.get("observed_on") or "",
                    "place": o.get("place_guess") or ""})
                added += 1
            print(f"{kind} ({name}): {have + added} photos", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
