-- Corn Pest Assessment (Cabadbaran) - Supabase database
--
-- Run this whole file once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Then run seed_data.sql. Running this file again is safe (nothing is duplicated).
-- See database/README.md for the full setup.
--
-- What is stored: who scanned, when, the photo, and the app's result (healthy / severity only /
-- pest + severity). No treatment advice or recommendations are stored anywhere.
-- Each user can only see and delete their own scans and photos (row level security).


-- ---------------------------------------------------------------------------------------------
-- Lookup tables (filled by seed_data.sql)
-- ---------------------------------------------------------------------------------------------

-- The five pests the app can name. "key" matches the pest classifier's class names.
create table if not exists public.pests (
  key             text primary key,
  english_name    text not null,
  scientific_name text not null
);

-- The 31 barangays of Cabadbaran City (PhilAtlas, November 2026).
create table if not exists public.barangays (
  id   smallint primary key,
  name text not null unique
);


-- ---------------------------------------------------------------------------------------------
-- User profiles: one row per user, created automatically when they sign up
-- ---------------------------------------------------------------------------------------------

create table if not exists public.profiles (
  id          uuid primary key references auth.users (id) on delete cascade,
  full_name   text,
  barangay_id smallint references public.barangays (id),
  created_at  timestamptz not null default now()
);

-- At sign-up the app sends {"full_name": "...", "barangay_id": "12"} as user metadata;
-- this copies it into profiles. An unknown barangay id is simply left empty.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, full_name, barangay_id)
  values (
    new.id,
    nullif(trim(new.raw_user_meta_data ->> 'full_name'), ''),
    (select b.id from public.barangays b where b.id::text = new.raw_user_meta_data ->> 'barangay_id')
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();


-- ---------------------------------------------------------------------------------------------
-- Scans: one row per saved result
-- ---------------------------------------------------------------------------------------------

-- Only real results are saved. "Not corn" and "unclear" photos are not saved: the app shows a
-- message and lets the user retake the photo.
do $$ begin
  create type public.scan_outcome as enum ('healthy', 'severity_only', 'pest');
exception when duplicate_object then null;
end $$;

-- "Unclear" is possible only with a pest: the insect was seen but no plant was found to measure.
do $$ begin
  create type public.severity_level as enum ('Healthy', 'Low', 'Moderate', 'High', 'Unclear');
exception when duplicate_object then null;
end $$;

create table if not exists public.scans (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null default auth.uid() references auth.users (id) on delete cascade,
  created_at      timestamptz not null default now(),
  photo_path      text not null,                    -- in the scan-photos bucket: <user id>/<file>.jpg
  outcome         public.scan_outcome not null,
  pest_key        text references public.pests (key),
  pest_confidence real check (pest_confidence between 0 and 1),
  damage_percent  real check (damage_percent between 0 and 100),
  severity        public.severity_level not null,
  barangay_id     smallint references public.barangays (id),  -- filled from the profile if empty
  model_version   text not null,                    -- which trained models gave this result
  details         jsonb not null default '{}'::jsonb,  -- every confidence from the backend, for analysis

  constraint pest_named_only_for_pest_outcome check ((outcome = 'pest') = (pest_key is not null)),
  constraint severity_fits_outcome check (
       (outcome = 'healthy'       and severity = 'Healthy')
    or (outcome = 'severity_only' and severity in ('Low', 'Moderate', 'High'))
    or (outcome = 'pest'          and severity in ('Low', 'Moderate', 'High', 'Unclear'))
  ),
  constraint photo_in_own_folder check (split_part(photo_path, '/', 1) = user_id::text)
);

create index if not exists scans_user_newest_first on public.scans (user_id, created_at desc);
create index if not exists scans_barangay_pest on public.scans (barangay_id, pest_key);

-- If the app leaves barangay_id empty, use the barangay from the user's profile.
create or replace function public.scan_default_barangay()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  if new.barangay_id is null then
    select p.barangay_id into new.barangay_id from public.profiles p where p.id = new.user_id;
  end if;
  return new;
end;
$$;

drop trigger if exists scans_default_barangay on public.scans;
create trigger scans_default_barangay
  before insert on public.scans
  for each row execute function public.scan_default_barangay();


-- ---------------------------------------------------------------------------------------------
-- Row level security: who may read and write what
-- ---------------------------------------------------------------------------------------------

alter table public.pests     enable row level security;
alter table public.barangays enable row level security;
alter table public.profiles  enable row level security;
alter table public.scans     enable row level security;

-- Pests and barangays: anyone may read them (the sign-up screen lists barangays before login).
drop policy if exists "anyone can read pests" on public.pests;
create policy "anyone can read pests" on public.pests
  for select to anon, authenticated using (true);

drop policy if exists "anyone can read barangays" on public.barangays;
create policy "anyone can read barangays" on public.barangays
  for select to anon, authenticated using (true);

-- Profiles: users read and update only their own (rows are created by the sign-up trigger).
drop policy if exists "read own profile" on public.profiles;
create policy "read own profile" on public.profiles
  for select to authenticated using (id = (select auth.uid()));

drop policy if exists "update own profile" on public.profiles;
create policy "update own profile" on public.profiles
  for update to authenticated using (id = (select auth.uid())) with check (id = (select auth.uid()));

-- Scans: users read, add and delete only their own. No update policy: a saved result can't be
-- edited.
drop policy if exists "read own scans" on public.scans;
create policy "read own scans" on public.scans
  for select to authenticated using (user_id = (select auth.uid()));

drop policy if exists "add own scans" on public.scans;
create policy "add own scans" on public.scans
  for insert to authenticated with check (user_id = (select auth.uid()));

drop policy if exists "delete own scans" on public.scans;
create policy "delete own scans" on public.scans
  for delete to authenticated using (user_id = (select auth.uid()));


-- ---------------------------------------------------------------------------------------------
-- Table rights for the app's two roles: anon (not logged in) and authenticated (logged in).
-- Newer Supabase projects don't grant these automatically ("permission denied for table ...");
-- the row level security rules above still decide which rows each user may touch.
-- ---------------------------------------------------------------------------------------------

grant usage on schema public to anon, authenticated;
grant usage on type public.scan_outcome, public.severity_level to anon, authenticated;
grant select on public.pests, public.barangays to anon, authenticated;
grant select, update on public.profiles to authenticated;
grant select, insert, delete on public.scans to authenticated;


-- ---------------------------------------------------------------------------------------------
-- Photo storage: a private bucket, one folder per user (<user id>/...)
-- ---------------------------------------------------------------------------------------------

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('scan-photos', 'scan-photos', false, 10485760, array['image/jpeg', 'image/png'])
on conflict (id) do nothing;

drop policy if exists "read own scan photos" on storage.objects;
create policy "read own scan photos" on storage.objects
  for select to authenticated
  using (bucket_id = 'scan-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "upload own scan photos" on storage.objects;
create policy "upload own scan photos" on storage.objects
  for insert to authenticated
  with check (bucket_id = 'scan-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "delete own scan photos" on storage.objects;
create policy "delete own scan photos" on storage.objects
  for delete to authenticated
  using (bucket_id = 'scan-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);
