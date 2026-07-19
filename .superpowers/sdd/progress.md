# OpenEnergy — İlerleme Defteri (SDD)

Branch: main (lokal-only, GitLab'a push YOK)

## Aşamalar
- [x] SP1: Veri temeli + provider  (plan: docs/superpowers/plans/2026-06-26-sp1-data-foundation.md)
- [x] SP2: Feature engineering
- [x] SP3: Experiment/AutoML motoru
- [x] SP4: Forecast serving + retrain
- [x] SP5: Frontend

## SP1 Task durumu
(implementer subagent uçtan uca yürütüyor; her task ayrı commit + pytest)

## Log

### SP1 review (sonnet) — bulgular
- Leakage boundary: SAĞLAM ✓
- Critical #1: previous_days paramı API'ye geçmiyor → FIX
- Important #2: _parse_previous_runs model suffix strip etmiyor → FIX
- Important #3: empty result all-String schema → FIX
- Important #4: lead_hours Int32→Int64 roundtrip drift → FIX
- Important #5: weather_raw uniqueness yok, re-ingest duplicate → FIX (UNIQUE + dedup)
- Minor #8: test_dataset_drops_rows_without_production yanlış path test ediyor → FIX
- Minor (defer/final): #6 dead retry code, #7 empty schema tutarsız, #9 schema-less concat, #10 model_seen non-deterministic

SP1: complete (12 task + 4 fix commits, review clean, 37 passed). Leakage boundary doğrulandı.

SP1 GERÇEKTEN complete: main fast-forward edildi 8a90cc6'ya, 37 passed doğrulandı.
DERS: subagent commit'leri detached kalabiliyor → her aşamadan sonra `git log -1` ile HEAD kontrolü şart.

SP2: complete (7 commits, main@b5d536d, 56 passed). Saf feature block'ları + registry + FeatureConfig + normalize.
SP2 fixes: 0140f6f (clear-sky gece guard, wind_dir 120m, get_block ValueError, dup validator), 60 passed.

SP3: implemented (10 commits, main@09744d4, 86 passed). Leakage izolasyonu opus-review ile SAĞLAM doğrulandı.
SP3 review C1 (critical): feature_blocks/nwp_source arama eksenleri atıl + champion config yalan → FIX.
SP3 fixes: 9ac8b9d (featurize axis real, I1 sentinel, M1 pinball NULL, M2 lgbm seed), 88 passed. SP3 COMPLETE.
  + I1 (CV atlanınca sentinel champion), M1 (ridge pinball NULL), M2 (lgbm seed).

SP4: implemented (6 commits, main@25d911b, 104 passed). Serving hizası SAĞLAM, champion/challenger doğru.
SP4 review: C1 (lgbm quantile crossing) FIX; I2 (temp file leak) FIX; I3 (empty feature_blocks → raise) FIX;
  M1 (temp dir leak) FIX; M3 (traceback) FIX; M2 (cross-experiment champion demote) FIX.
  I1 (forecast lead NULL covariate shift): reviewer literal fix YANLIŞ (lead NULL) → dokümante edilecek, tek-güncel-forecast varsayımı.

SP4 fixes: dc4b8b3 (C1 monotonicity + I2/I3 + M1/M2/M3 + I1 doc), 108 passed. SP4 COMPLETE.

## SP5 Tasks

Task 0: complete (commits 57fa5a7..d43c8b3, review clean) — backend: expose experiment_id via job detail
Task 1: complete (commits d43c8b3..421ac96, review clean) — Next.js scaffold + Tailwind + deps
Task 2: complete (commits 421ac96..591db39, review clean) — typed API client + types + tests (3/3 passing)
Task 3: complete (commits 591db39..923e635, review clean) — PlantForm + home page
Task 4: complete (commits 923e635..46b313e, review clean) — MapLibre point map (SSR-safe, next/dynamic)
Task 5: complete (commits 46b313e..5cbdae6, review clean) — ProductionUpload CSV component
Task 6: complete (commits 5cbdae6..6e87857, review clean) — ExperimentPanel (start/poll/results, skill<0 red warning). Minor: nullable key={point_id} falls back to undefined — cosmetic.
Task 7: complete (commits 6e87857..4356da6, review clean) — ForecastChart (Recharts p10/p50/p90) + plant detail page (4 sections)
Task 8: complete (commits 4356da6..f64f20d, review clean) — frontend/README.md + root README.md with --factory flag

SP5: COMPLETE (main@076a0d1). 10 commits total (Task 0 backend + Tasks 1-8 frontend + I-1 fix). Build: ✓, tests: 3/3, backend: 109 passed.

SP5: complete (10 commits, main@076a0d1). Frontend build PASS, 3/3 client test, backend 109 passed. node_modules/.next ignored.
TÜM AŞAMALAR (SP1-SP5) TAMAMLANDI.

FINAL review (sonpsnot): CORS Critical → 67f44a3 ile düzeltildi, 110 passed. Canlı Open-Meteo entegrasyon testi GEÇTİ.
Kalan Minor (kabul, lokal MVP): forecast chart fill kozmetik; job detail traceback (lokal-only); timezone test boşluğu.
DURUM: SP1-SP5 TAMAM. Lokal main. GitHub URL bekleniyor (push kullanıcı onayıyla).
