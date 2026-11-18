-- Corn Pest Assessment (Cabadbaran) - lookup data
--
-- Run after schema.sql (Supabase dashboard: SQL Editor -> New query -> paste -> Run).
-- Running it again is safe: existing rows are updated, nothing is duplicated.
-- Names only: no treatment advice or recommendations.

-- The five pests. "key" must match the pest classifier's class names (ml/src/train.py).
insert into public.pests (key, english_name, scientific_name) values
  ('fall_armyworm',    'Fall armyworm',    'Spodoptera frugiperda'),
  ('common_cutworm',   'Common cutworm',   'Spodoptera litura'),
  ('armyworm',         'Armyworm',         'Mythimna separata'),
  ('corn_planthopper', 'Corn planthopper', 'Peregrinus maidis'),
  ('corn_aphid',       'Corn aphid',       'Rhopalosiphum maidis')
on conflict (key) do update
  set english_name = excluded.english_name, scientific_name = excluded.scientific_name;

-- The 31 barangays of Cabadbaran City, Agusan del Norte
-- (source: PhilAtlas, https://philatlas.com/mindanao/caraga/agusan-del-norte/cabadbaran.html).
-- Ids are fixed so saved scans keep pointing at the right barangay.
insert into public.barangays (id, name) values
  ( 1, 'Antonio Luna'),
  ( 2, 'Bayabas'),
  ( 3, 'Bay-ang'),
  ( 4, 'Caasinan'),
  ( 5, 'Cabinet'),
  ( 6, 'Calamba'),
  ( 7, 'Calibunan'),
  ( 8, 'Comagascas'),
  ( 9, 'Concepcion'),
  (10, 'Del Pilar'),
  (11, 'Katugasan'),
  (12, 'Kauswagan'),
  (13, 'La Union'),
  (14, 'Mabini'),
  (15, 'Mahaba'),
  (16, 'Poblacion 1'),
  (17, 'Poblacion 2'),
  (18, 'Poblacion 3'),
  (19, 'Poblacion 4'),
  (20, 'Poblacion 5'),
  (21, 'Poblacion 6'),
  (22, 'Poblacion 7'),
  (23, 'Poblacion 8'),
  (24, 'Poblacion 9'),
  (25, 'Poblacion 10'),
  (26, 'Poblacion 11'),
  (27, 'Poblacion 12'),
  (28, 'Puting Bato'),
  (29, 'Sanghan'),
  (30, 'Soriano'),
  (31, 'Tolosa')
on conflict (id) do update set name = excluded.name;
