# Database (Supabase)

Supabase gives the app three things: **login**, a **database** for scan results, and **photo
storage**. No treatment advice or recommendations are stored anywhere.

| File | What it does |
|---|---|
| [schema.sql](schema.sql) | Tables, security rules, and the private `scan-photos` photo folder |
| [seed_data.sql](seed_data.sql) | The 5 pests and the 31 barangays of Cabadbaran City |

## How a scan is saved

```
App: user logs in (email + password)
 -> takes a photo -> sends it to the FastAPI backend -> backend runs the 4 models -> result
 -> app uploads the photo to scan-photos/<user id>/<scan id>.jpg
 -> app saves the result as a row in "scans"
 -> History screen lists the user's own scans, newest first
```

The backend never connects to the database, so no secret database key is needed on the server.
The app uses only Supabase's public key; the security rules decide what each user may do.

## Tables

| Table | Holds |
|---|---|
| `pests` | The 5 pests: key, English name, scientific name |
| `barangays` | The 31 barangays (fixed ids 1-31) |
| `profiles` | One row per user: full name, barangay. Created automatically at sign-up |
| `scans` | One row per saved result: user, date/time, photo path, outcome (`healthy` / `severity_only` / `pest`), pest, pest confidence, damage %, severity (`Healthy` / `Low` / `Moderate` / `High`, or `Unclear` for a pest with no plant to measure), barangay, model version, and every confidence in `details` |

"Not corn" and "unclear" photos are **not** saved: the app shows a message and lets the user
retake the photo.

**Security rules (row level security):**

- Each user can read, add and delete only their own scans and photos, and read and edit only
  their own profile.
- A saved result can't be edited. Inconsistent results are refused:
  - a pest name without the pest outcome;
  - "Healthy" with a damage grade;
  - a confidence above 1;
  - a photo outside the user's own folder.
- Anyone, even before login, can read the pest and barangay lists, so the sign-up screen can
  show the barangays.
- If the app saves a scan without a barangay, the user's profile barangay is used.

## Setup (once, about 10 minutes)

1. **Create a project.** Go to [supabase.com](https://supabase.com), sign in, and create a new
   project.
   - Name: for example `corn-pest`.
   - Region: Southeast Asia (Singapore), the closest to Cabadbaran.
   - Choose a database password and keep it private. The app doesn't need it.
2. **Create the tables.** Open **SQL Editor** and click **New query**. Paste all of
   `schema.sql` and click **Run**. Then do the same with `seed_data.sql`. Both files are safe to
   run again if something goes wrong halfway.
3. **Check it worked.**
   - **Table Editor** shows `pests` (5 rows), `barangays` (31 rows), `profiles` and `scans`
     (empty).
   - **Storage** shows a private bucket called `scan-photos`.
4. **Turn on email login.** In **Authentication**, under sign-in methods / providers, make sure
   **Email** is on. While testing you may turn off "Confirm email", so test accounts work
   without opening a confirmation email. Turn it back on before real farmers use the app.
5. **Copy two values for the app.** In **Project Settings -> API** (or **API Keys**), copy:
   - the **Project URL**;
   - the **anon / publishable key**.

   These two go into the Flutter app in step 9. They are public and safe in the app because of
   the security rules.

   **Never** put the `service_role` / secret key in the app or share it with anyone, including
   in chat.

If a query shows an error, copy the error message and the line number, and we'll fix it.

## Saving a scan from the app (for step 9)

The backend's result (`ml/src/flow.py`) maps to the `scans` columns like this:

| Backend result | `scans` column |
|---|---|
| `outcome` (`healthy`, `severity_only`, `pest`) | `outcome`. Don't save `not_corn` or `unclear` |
| `pest` | `pest_key` (only when the outcome is `pest`) |
| `pest_confidence` | `pest_confidence` |
| `damage_percent` | `damage_percent` |
| `severity` | `severity` |
| the backend's model version | `model_version` |
| all the confidences | `details` (JSON) |
| the uploaded photo's path, `<user id>/<scan id>.jpg` | `photo_path` |

To delete a scan, delete both its photo in `scan-photos` and its row in `scans`.

At sign-up the app sends the name and barangay as user metadata:
`{"full_name": "Juan Dela Cruz", "barangay_id": "13"}`. The profile is then filled in
automatically.

## Data for the paper

In the SQL Editor (as the project owner you see every user's scans), pests found per barangay:

```sql
select b.name as barangay, coalesce(p.english_name, s.outcome::text) as result, count(*) as scans
from scans s
left join barangays b on b.id = s.barangay_id
left join pests p on p.key = s.pest_key
group by 1, 2
order by 1, 3 desc;
```

## How this was tested

Both files were run (twice each) on PostgreSQL 18, with a small stand-in for Supabase's login
and storage. Like newer Supabase projects, the stand-in gives the app's roles no table rights
of its own, so the `grant` lines in `schema.sql` alone must be enough. (They were missing at
first, which showed as "permission denied for table barangays" on a real project, 20 Nov 2026.)
Checks that passed:

- **Sign-up and profiles.** Sign-up creates the profile with the chosen barangay, and an
  unknown barangay is left empty.
- **Before login.** Someone not logged in can read the barangays and pests, but no profiles or
  scans.
- **Saving scans.** A user can save scans, and the profile barangay is filled in when the app
  leaves it empty.
- **Bad scans refused.** All 7 of these were refused:
  - a scan for another user;
  - "Healthy" with "Low";
  - a pest outcome without a pest name;
  - a pest name without the pest outcome;
  - a photo in another user's folder;
  - an unknown pest;
  - a confidence above 1.
- **Editing.** Saved results can't be edited.
- **Photos.** A user can only add photos to their own folder.
- **Privacy.** Another user sees none of the first user's scans, photos or profile, and can't
  delete them.

Supabase's real login and storage weren't part of this test, so check step 3 above after the
setup.
