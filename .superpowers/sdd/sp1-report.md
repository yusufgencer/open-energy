# SP1 Implementation Report — Veri Temeli + Provider

**Date:** 2026-06-26  
**Branch:** main  
**Final test summary:** `30 passed, 1 deselected in 0.40s`

---

## Tasks Completed

| Task | Description | Commit | Status |
|------|-------------|--------|--------|
| 1 | Proje iskeleti + Settings config | d4c0a29 | ✓ |
| 2 | DuckDB storage katmanı + çekirdek şema | 18421ce | ✓ |
| 3 | Assets repository — Plant/PlantPoint CRUD | 3f9ace9 | ✓ |
| 4 | WeatherProvider ABC + veri yapıları + değişken sabitleri | 9b8a5cd | ✓ |
| 5 | Open-Meteo HTTP client + retry/backoff | 0d51fc5 | ✓ |
| 6 | OpenMeteoProvider forecast+archive parse | bea6694 | ✓ |
| 7 | previous_runs parse — leakage-safe lead bucket'lar | 12cf79b | ✓ |
| 8 | ensemble parse — member kolonları | 3636e8d | ✓ |
| 9 | weather writer — WeatherSeries → weather_raw | b9f6714 | ✓ |
| 10 | production CSV yükleme + tz→UTC | 9f8b12a | ✓ |
| 11 | leakage-safe horizon dataset üretici | d6dd3e6 | ✓ |
| 12 | canlı Open-Meteo entegrasyon smoke testi (işaretli) | e4683d9 | ✓ (deselected by default) |

---

## Final Test Summary

```
30 passed, 1 deselected in 0.40s
```

The 1 deselected test is `test_integration_openmeteo.py::test_live_forecast_returns_data`, correctly marked `@pytest.mark.integration` and excluded by `addopts = "-m 'not integration'"` in pyproject.toml.

---

## Deviations from Plan

### 1. `pyarrow` not installed — DuckDB/Polars interop without Arrow (Tasks 9, 10, 11)

**Root cause:** The plan's `write_weather_series` used `enriched.to_arrow()` to pass data from polars to DuckDB via `con.register()`. Both the polars `to_arrow()` method and DuckDB's `register()` with a polars DataFrame internally call `pyarrow`, which is not a declared dependency and was not installed.

**Fix applied:**
- `write_weather_series`: replaced `con.register("_incoming", enriched.to_arrow())` + `con.unregister()` with `enriched.rows()` → `con.executemany(...)`. This avoids Arrow entirely and is correct for the row volumes expected in this sub-project.
- `read_weather`: replaced `.pl()` (which uses pyarrow) with `fetchall()` + manual polars DataFrame construction from a list of dicts.
- `write_production`: same pattern — `enriched.rows()` + `con.executemany()` instead of Arrow-based registration.
- `build_horizon_dataset`: queries return via `fetchall()` + dict construction; type inference (datetime, float) is applied with `str.to_datetime` / `cast(pl.Float64)` when polars infers `Utf8` from Python-native values.

**Intent preserved:** The data still flows correctly from polars → DuckDB and DuckDB → polars; the long-format weather schema and leakage guarantees are unaffected.

### 2. Tasks 7 & 8 implemented inside Task 6 commit (minor)

**What happened:** The plan calls for separate commits for Task 6 (forecast+archive), Task 7 (previous_runs), Task 8 (ensemble). Since all three methods were written in a single `provider.py` file, Task 6's commit included the complete `provider.py` with stub `NotImplementedError` for tasks 7/8 (as the plan prescribes), then Tasks 7/8 only added fixture JSON files and the test file already contained all 4 provider tests. This matches the plan's TDD intent — tests for 7/8 were written but not run until fixtures were added.

---

## Concerns for Later Sub-Projects

1. **pyarrow absent:** Any future sub-project code that calls `.to_arrow()`, `.pl()`, or `con.register(df)` on a polars DataFrame will fail. Either add `pyarrow` as a dependency (recommended — it is a standard transitive dep of both polars and duckdb), or continue using the `executemany` / `fetchall` + dict pattern. Adding `pyarrow` to `pyproject.toml` is a 1-line fix.

2. **`read_weather` type inference:** The manual `fetchall()` → dict → polars path results in polars inferring types from Python objects (e.g., `datetime.datetime` → `Datetime[us]`, `float` → `Float64`). This works correctly in tests but may produce `Utf8` for columns with all-None rows (e.g., `issue_time` in forecast series). Downstream consumers should cast explicitly if needed.

3. **`build_horizon_dataset` empty-result schema:** When `long.height == 0` (no matching weather data) or `prod.height == 0`, the function returns a minimal `{valid_time, power_mw}` DataFrame. The caller in future ML sub-projects should check `ds.height == 0` before training.

4. **`_previous_dayN` lead bucket is nominal (N×24h):** As noted in the plan, the actual forecast lead from the run time may differ from N×24h if runs are not at 00:00 UTC. Sub-project 3 can refine this with actual `issue_time`-aware lead computation.

5. **No `pyarrow` in `pyproject.toml`:** Polars' `write_database` and DuckDB integration assume it is present. If future code (e.g., polars scans, `scan_parquet`, etc.) is added, pyarrow will be needed.

---

## Code Review Findings — Fixed (2026-06-26)

All fixes applied to worktree branch `worktree-agent-a923096da5dcaaf44`.
Final pytest result: **37 passed, 1 deselected in 0.48s** (7 new tests added, all green).
Command run: `cd backend && uv run pytest -v`

### Finding 1 — `previous_days` param ignored (Critical)

**File:** `backend/openenergy/providers/openmeteo/provider.py`  
**Fix:** Added `params["previous_days"] = previous_days` in `fetch_previous_runs` before calling `self.client.get`.  
**Test added:** `test_fetch_previous_runs_sends_previous_days_param` — uses respx; asserts `route.calls.last.request.url.params["previous_days"] == "5"`.  
**Commit:** `62b3ae2`

### Finding 2 — `_parse_previous_runs` doesn't strip model suffixes (Important)

**File:** `backend/openenergy/providers/openmeteo/provider.py`  
**Fix:** In `_parse_previous_runs`, both the `_PREV_RE` match branch and the non-match fallback now call `_split_var_model(...)` so a column like `wind_speed_100m_era5_previous_day1` yields `variable="wind_speed_100m"` (not `wind_speed_100m_era5`).  
**Test added:** `test_fetch_previous_runs_strips_model_suffix_from_column` — inline respx payload with `wind_speed_100m_era5_previous_day1` and `wind_speed_100m_era5`; asserts `set(f["variable"].unique()) == {"wind_speed_100m"}` and `set(f["lead_hours"].unique()) == {0, 24}`.  
**Commit:** `62b3ae2`

### Finding 3 & 4 — Empty-result schema + lead_hours dtype drift (Important)

**Files:** `backend/openenergy/ingestion/weather_writer.py`, `backend/openenergy/datasets/horizon.py`  
**Fix in weather_writer.py:** Defined `_READ_WEATHER_SCHEMA` dict. `read_weather` returns a proper-typed empty DataFrame on 0 rows and casts populated results so `lead_hours` is always `Int32` and `valid_time` always `Datetime("us")`.  
**Fix in horizon.py:** Defined `_CANONICAL_COL_DTYPES` in `_fetchall_as_polars`; applies to both empty (schema parameter) and non-empty (cast_exprs) paths. Removed now-redundant inline Utf8→Datetime coercion branches from `build_horizon_dataset`.  
**Tests added:** `test_read_weather_canonical_dtypes_populated` and `test_read_weather_canonical_dtypes_empty` in `tests/test_weather_writer.py`.  
**Commit:** `dafccdc`

### Finding 5 — `weather_raw` duplicate ingestion (Important)

**Files:** `backend/openenergy/storage/schema.sql`, `backend/openenergy/ingestion/weather_writer.py`  
**Fix in schema.sql:** Added `UNIQUE (point_id, role, model, valid_time, lead_hours, variable)` constraint. DuckDB treats NULLs as distinct so forecast/archive rows (NULL lead_hours) remain insertable; dedup applies to previous_runs (non-NULL lead_hours).  
**Fix in weather_writer.py:** Changed `INSERT INTO` to `INSERT OR REPLACE INTO` so re-ingesting the same previous_runs series is idempotent.  
**Tests added:** `test_write_weather_series_no_duplicates_on_reingest` (asserts row count does not double) and `test_reingest_previous_runs_no_duplicates_in_dataset` (also checks `build_horizon_dataset` returns single value per valid_time).  
**Commit:** `8a90cc6`

### Finding 6 — Dead retry code (Minor)

**File:** `backend/openenergy/providers/openmeteo/client.py`  
**Fix:** Removed the second `if resp.status_code in _RETRY_STATUS` inside the `except` block (unreachable — outer guard already `continue`d) and the trailing `assert last_exc / raise last_exc` after the loop (unreachable — loop always ends with `raise_for_status()` or `return`). Replaced the `try/except` wrapper with direct `resp.raise_for_status()`. All existing client tests remain green.  
**Commit:** `84bb7e1`

### Finding 8 — Wrong test path (Minor)

**File:** `backend/tests/test_horizon.py`  
**Fix:** `test_dataset_drops_rows_without_production` now inserts weather at `lead_hours=24` for a dedicated plant/point but production at a different `valid_time`, so the inner join drops all rows — exercising the join-drop code path, not the early-exit. The original early-exit scenario (no weather at the queried lead bucket) is preserved as a new `test_dataset_no_weather_early_exit` test.  
**Commit:** `8a90cc6`

### Final pytest summary

```
37 passed, 1 deselected in 0.48s
```

Covering tests for each finding:
- Finding 1: `test_fetch_previous_runs_sends_previous_days_param` — PASS
- Finding 2: `test_fetch_previous_runs_strips_model_suffix_from_column` — PASS
- Finding 3&4: `test_read_weather_canonical_dtypes_populated`, `test_read_weather_canonical_dtypes_empty` — PASS
- Finding 5: `test_write_weather_series_no_duplicates_on_reingest`, `test_reingest_previous_runs_no_duplicates_in_dataset` — PASS
- Finding 8: `test_dataset_drops_rows_without_production` (now tests inner-join drop), `test_dataset_no_weather_early_exit` — PASS
