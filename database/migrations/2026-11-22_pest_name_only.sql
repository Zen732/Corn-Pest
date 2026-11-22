-- Update a database made before 22 Nov 2026 (run once in Supabase: SQL Editor -> New query -> Run).
-- Safe to run again.
--   * new outcome "unknown_pest": an insect was found but the app can't tell which pest it is
--   * a pest result now has no severity (the pest's name only); older pest scans keep theirs

alter type public.scan_outcome add value if not exists 'unknown_pest';

alter table public.scans alter column severity drop not null;

alter table public.scans drop constraint if exists severity_fits_outcome;
alter table public.scans add constraint severity_fits_outcome check (
    case outcome::text
      when 'healthy'       then coalesce(severity::text, '') = 'Healthy'
      when 'severity_only' then coalesce(severity::text, '') in ('Low', 'Moderate', 'High')
      when 'pest'          then severity is null or severity::text in ('Low', 'Moderate', 'High', 'Unclear')
      when 'unknown_pest'  then severity is null
      else false
    end
  );
