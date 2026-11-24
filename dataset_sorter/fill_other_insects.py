"""Version 2: photos of insects (and spiders) that are NOT one of the five corn pests, for an
"other insect" answer in the pest classifier, so an unknown insect is not named as a corn pest.

Why: a beetle on a corn leaf was named "Corn aphid, 87%" (22 Nov 2026). The pest classifier only
knew the five pests, so it had to pick one of them.

Adds ml/dataset/raw/corn/other_insect/ (nothing existing is changed). iNaturalist research-grade
observations with open licences; at most PER_SPECIES photos per species, newest first. Left out,
so no photo shows one of the five pests or a close look-alike: the genera Spodoptera (fall
armyworm, common cutworm), Mythimna (armyworm), Rhopalosiphum (corn aphid), Peregrinus (corn
planthopper), and the whole families Aphididae (aphids) and Delphacidae (planthoppers).

    group             photos  where
    caterpillars       1,000  Asia (only 409 larva photos exist for the Philippines)
    beetles              600  Philippines
    true_bugs            500  Philippines
    grasshoppers         400  Philippines
    other_insects        700  Philippines, any insect
    spiders              300  Philippines
  round 2 (23 Nov 2026, after the web test: other moths were named armyworm / fall armyworm, and
  whiteflies / mites corn aphid):
    adult_moths        1,000  Asia, adult moths and butterflies
    small_sucking        400  Asia: whiteflies, mealybugs, scale insects, thrips, psyllids
    leafhoppers          300  Asia (look like planthoppers, but are not)
    (mites               150  Asia: archived again, mostly leaf galls with no animal in view)
  Round 2 also leaves out the 25 "other pest" species of the web test (fill_web_test.py) and its
  observations, so that test stays a test of insects the app never saw.

Credited in ml/dataset/inaturalist_credits.csv (backed up first). iNaturalist is asked about once
per second with the project's generic User-Agent. The observation lists are saved in
dataset_sorter/downloads/ and reused. Safe to run again: it tops up.

Run from the project root:
    dataset_sorter/.venv/Scripts/python.exe dataset_sorter/fill_other_insects.py
"""
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fill_real_world import backup, known_hashes, pick_spread
from inat_download import (API, CREDIT_FIELDS, CREDITS_FILE, LICENSES, RAW_DIR, append_csv, fetch_photo, get_json,
                           is_duplicate, list_images)

OUT = RAW_DIR / "corn" / "other_insect"
LISTS = Path(__file__).resolve().parent / "downloads"
PHILIPPINES, ASIA = 6873, 97395
LEAVE_OUT = "118931,121184,142772,330086,52381,123608"  # Spodoptera, Mythimna, Rhopalosiphum, Peregrinus, Aphididae, Delphacidae
PER_SPECIES = 2
ADULT = {"term_id": 1, "term_value_id": 2}
# The web test's other pests (fill_web_test.py), left out from round 2 on.
WEB_TEST = ("490172,83806,126276,141725,145555,199416,911325,124869,358125,904700,201627,493892,124811,52045,"
            "320058,130081,336925,354799,335537,55626,51583,203333,922656,305001,196542")
GROUPS = [  # name, query, target, pages to list (200 observations each)
    ("caterpillars", {"taxon_id": 47157, "term_id": 1, "term_value_id": 6, "place_id": ASIA}, 1000, 40),
    ("beetles", {"taxon_id": 47208, "place_id": PHILIPPINES}, 600, 20),
    ("true_bugs", {"taxon_id": 47744, "place_id": PHILIPPINES}, 500, 15),
    ("grasshoppers", {"taxon_id": 47651, "place_id": PHILIPPINES}, 400, 15),
    ("other_insects", {"taxon_id": 47158, "place_id": PHILIPPINES}, 700, 40),
    ("spiders", {"taxon_id": 47119, "place_id": PHILIPPINES}, 300, 25),
    ("adult_moths", {"taxon_id": 47157, **ADULT, "place_id": ASIA, "leave_out_web": 1}, 1000, 40),
    ("small_sucking", {"taxon_id": "154784,49574,60922,83201,48195", "place_id": ASIA, "leave_out_web": 1}, 400, 25),
    ("leafhoppers", {"taxon_id": 53237, "place_id": ASIA, "leave_out_web": 1}, 300, 20),
    # mites (Acari, Asia, 150) were downloaded and then archived (_archive/2_unused_photos/
    # other_insect_mites_leaf_galls/): most photos show leaf galls, not a visible animal.
]


def candidates(name, query, pages):
    f = LISTS / f"inat_other_insects_{name}.json"
    if f.exists():
        return [tuple(x) for x in json.loads(f.read_text(encoding="utf-8"))]
    query = dict(query)
    leave_out = LEAVE_OUT + ("," + WEB_TEST if query.pop("leave_out_web", 0) else "")
    params = {**query, "without_taxon_id": leave_out, "quality_grade": "research", "photos": "true",
              "photo_license": ",".join(sorted(LICENSES)), "order_by": "id", "order": "desc", "per_page": 200}
    out, id_below = [], None
    for _ in range(pages):
        q = dict(params, **({"id_below": id_below} if id_below else {}))
        page = get_json(f"{API}?{urllib.parse.urlencode(q)}")["results"]
        time.sleep(1)  # about 1 request per second
        if not page:
            break
        for obs in page:
            photo = next((p for p in obs.get("photos", []) if p.get("license_code") in LICENSES and p.get("url")), None)
            taxon = obs.get("taxon") or {}
            if photo and taxon.get("id"):
                out.append(({"id": obs["id"], "taxon": {"id": taxon["id"], "name": taxon.get("name", "")},
                             "observed_on": obs.get("observed_on"), "place_guess": obs.get("place_guess")},
                            {k: photo.get(k) for k in ("url", "license_code", "attribution")}))
        id_below = page[-1]["id"]
        print(f"  {name}: {len(out)} observations listed", end="\r", flush=True)
    LISTS.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


def credited_ids():
    """Observations already in the dataset, or in the web test (never trained on)."""
    import csv
    ids = set()
    for f in (CREDITS_FILE, CREDITS_FILE.parent / "test_web_credits.csv"):
        if f.exists():
            with open(f, newline="", encoding="utf-8") as fh:
                ids |= {r["observation_id"] for r in csv.DictReader(fh)}
    return ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    backup(CREDITS_FILE)
    known = known_hashes([OUT])
    used = credited_ids()  # never reuse an observation already in the dataset (any folder)
    for name, query, target, pages in GROUPS:
        have = sum(1 for f in list_images(OUT) if f.name.startswith(f"{name}_"))
        need = target - have
        print(f"\n{name}: {have} already, {max(0, need)} to add", flush=True)
        if need <= 0:
            continue
        picks = [(o, p) for o, p in pick_spread(candidates(name, query, pages), None, PER_SPECIES)
                 if str(o["id"]) not in used]
        print(f"\n  {len(picks)} usable picks", flush=True)
        added = 0
        with ThreadPoolExecutor(4) as pool:
            for i in range(0, len(picks), 16):
                if added >= need:
                    break
                chunk = picks[i: i + 16]
                got = pool.map(fetch_photo, [p["url"].replace("/square.", "/medium.") for _, p in chunk])
                for (obs, photo), res in zip(chunk, got):
                    if res is None or added >= need:
                        continue
                    img, h = res
                    if is_duplicate(h, known):
                        continue
                    known.append(h)
                    used.add(str(obs["id"]))
                    fname = f"{name}_inat_{obs['id']}.jpg"
                    img.save(OUT / fname, quality=92)
                    append_csv(CREDITS_FILE, CREDIT_FIELDS, {
                        "file": f"corn\\other_insect\\{fname}", "observation_id": obs["id"],
                        "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                        "species": obs["taxon"].get("name", ""), "license": photo["license_code"],
                        "attribution": photo.get("attribution", ""), "observed_on": obs.get("observed_on") or "",
                        "place": obs.get("place_guess") or ""})
                    added += 1
                print(f"  added {added}/{need}", end="\r", flush=True)
                time.sleep(1)
        print(f"\n  {name}: added {added}")
    print(f"\nother_insect: {len(list_images(OUT))} photos")


if __name__ == "__main__":
    main()
