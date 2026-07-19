# OpenEnergy — Proje Durumu ve Devam Notu

_Son güncelleme: 2026-07-16_

Bu doküman, çalışmanın nerede kaldığını ve sıradaki adımları kaydeder. Yeni bir
oturuma başlarken önce bunu oku.

## Genel mimari (kurulu ve çalışıyor)

- **Backend**: Python + FastAPI + **kalıcı DuckDB** (`data/openenergy.duckdb`, gitignore'lu).
  Port **8010**. Başlatma:
  ```bash
  cd backend && nohup uv run uvicorn openenergy.api.app:create_app --factory \
    --host 127.0.0.1 --port 8010 > /tmp/openenergy-backend.log 2>&1 & disown
  ```
  Not: DuckDB tek-yazar/proses → tüm ingestion API endpoint'lerinde in-process yapılır.
- **Frontend**: Next.js 16 + React 19 + Tailwind v4 + shadcn (nova). Port **3000**.
  `frontend/.env.local` → `NEXT_PUBLIC_API_URL=http://127.0.0.1:8010`.
- Testler: backend `cd backend && uv run pytest -q` (**384 passed**, ~3 dk — pvlib/conformal
  ile büyüdü), frontend `cd frontend && npx tsc --noEmit && npx vitest run` (**11 passed**).
- Git: lokal `main` + feature branch **`feature/forecasting-pool`** (tüm strateji-havuzu işi
  burada, 37 commit). **Push YOK** (remote yok; kullanıcı onaylayınca push/merge).

## Strateji Havuzu Programı — TAMAMLANDI (Faz A→F)

Amaç: her santral için **en başarılı yöntemi keşfettiren aranabilir havuz** + onu seçen
arama/hafıza/değerlendirme makinesi. Yol haritası ve tam görev listesi:
`docs/superpowers/plans/2026-07-16-forecasting-roadmap.md` ve
`docs/superpowers/plans/2026-07-16-forecasting-program-tasklist.md`. SOTA araştırma sentezi:
`docs/superpowers/specs/2026-07-16-sota-forecasting-research-synthesis.md`.

Tüm fazlar TDD ile uygulandı ve commit'lendi (çoğu bir görev = bir subagent):

- **Faz A — Temeller**: merkezi quantile monotonluk (`enforce_quantile_order`, clamp-to-p50);
  `diurnal_persistence` baseline + valid_time-farkındalıklı skill; solar `peak_bias/peak_mae`
  + `crps_approx`; coverage/width/crps/peak kolonları experiment_trials/strategy_trials/
  scenario_runs'a (idempotent migration); results/scenario API + frontend tipleri.
- **Faz B — Çok-nokta + çok-kaynak**: chunk'lı bulk weather writer; `nwp_trajectory` +
  `wind_physics_ext` blokları; `features/spatial.py` (aggregate_points, SpatialPCA, halkalar);
  NWP-native **grid üretici** (`ingestion/grid.py`); **point_policy** ekseni
  {single_best, spatial_mean, mean_plus_spread, all_points} + **source_policy**
  {ecmwf, icon_eu, gfs, combo}; multipoint assembly + runner araması + **serving parity**;
  scenario memory'de gerçek policy kaydı; policy leaderboard; API + **UI dikey**.
- **Faz C — Solar tepe-yakalama**: **pvlib** (`physics/solar.py` clear-sky power); solar
  geometry + `pv_model_chain` blokları; **kPV target_policy** (P/P_cs, geri çarp); gündüz
  maskeleme + gece sıfır; P_cs-orantılı sample weight; target_policy scenario memory'de kanıtlı.
- **Faz D — Olasılıksal kalite**: full-quantile validation matrisleri; **per-quantile ensemble**
  (Vincentization + QRA meta); ensemble bantları uçtan uca korunur; **conformal** CQR/ACI
  (`calibration/conformal.py`) + **quantile_policy** ekseni; **CatBoost MultiQuantile**,
  RF→**QRF**, **seed-bagging** havuz üyeleri.
- **Faz E — Operasyonel**: **Diebold-Mariano** anlamlılık kapılı champion-challenger; aylık
  retrain endpoint/job; günlük **LASSO polinom quantile post-processor**; **drift monitor** +
  drift-tetikli acil retrain; retrain sonuçları scenario memory'de.
- **Faz F — Mimari**: previous_day3–7 ile **72–168h horizon** + skill kapısı; **pooled GBM +
  lead_time** rakibi; **Open-Meteo Single Runs API** (config-gated, **KAPALI varsayılan**,
  ücretli); **dinamik reforecast** + time-lagged run blending (`forecasting/reforecast.py`).

## Gerçek app'te doğrulandı (2026-07-16)

Backend ayağa kaldırıldı; uçtan-uca akış canlı koşturuldu:
- Tüm yeni endpoint'ler kayıtlı (weather-grid, reforecast, retrain, postprocessor, drift,
  scenario/policy-leaderboard, experiments/{id}/champion). App tüm yeni modülleri (pvlib,
  conformal, retraining, reforecast) sorunsuz import ediyor.
- Plant + turbine point + **3×3 icon_eu grid** (native 0.0625° aralık, merkez tam noktada, 9 nokta).
- **Canlı Open-Meteo çok-noktalı ingest**: 9 noktaya **57.024 satır** (previous-runs, lead 0/24/48).
- Sentetik üretim yüklendi (1176 satır) → **çok-noktalı deney**: point_policy araması
  `single_best/spatial_mean/all_points`'i keşfedip stacking_meta_ridge ile birleştirdi;
  `source_policy=icon_eu`; strategy_family policy'leri içeriyor. **Skill = -0.19** (üretim
  sentetik/gerçek-hava-ile-ilişkisiz olduğundan doğru şekilde negatif — A2 skill makinesi dürüst).
- Champion + **policy leaderboard** endpoint'leri kazanan nokta/kaynak politikasını dönüyor.
- **reforecast** endpoint'i çalıştı (`fired=True, rows=0` — serving için forecast-role hava
  ingest edilmediğinden 0; beklenen davranış).

## Gerçek RES veri operasyonu — TAMAMLANDI (2026-07-16)

Orijinal "SIRADAKİ İŞ" gerçek veriyle uçtan uca koşturuldu: 12 orta-kapasiteli (30-63 MW)
Türk RES'i seçildi, koordinatlar GEM/EPDK + enerjiatlasi'den doğrulanıp EPİAŞ id'leriyle
eşleştirildi, her biri için plant + türbin noktası + 3×3 icon_eu grid kuruldu, Open-Meteo
previous-runs hava (her plant 238.464 satır) + **gerçek EPİAŞ realtime-generation üretim**
(her plant 1104 satır, 46 gün) ingest edildi, çok-noktalı deney (horizon 24/48) koşuldu.

**Sonuç: 24 champion (12 RES × 2 horizon), 24'ü de POZİTİF skill (ortalama 0.273).**

| RES | MW | h24 skill | h48 skill |
|---|---|---|---|
| yuntdag | 57.5 | 0.467 | 0.384 |
| canta | 50.0 | 0.463 | 0.447 |
| seyitali | 40.7 | 0.420 | 0.390 |
| belen | 48.0 | 0.320 | 0.259 |
| camseki | 63.1 | 0.293 | 0.294 |
| akbuk | 31.5 | 0.286 | 0.292 |
| kozbeyli | 34.55 | 0.250 | 0.214 |
| sayalar | 57.2 | 0.207 | 0.169 |
| sebenoba | 60.0 | 0.175 | 0.201 |
| mazi3 | 30.0 | 0.172 | 0.187 |
| intepe | 55.7 | 0.158 | 0.201 |
| kuyucak | 50.1 | 0.139 | 0.173 |

- Skill = diurnal/persistence baseline'a karşı iyileşme; **hepsi pozitif** → motor gerçek EPİAŞ
  üretimini gerçek hava ile baseline'dan %14-47 daha iyi tahmin ediyor.
- Kazanan point_policy tutarlı olarak **`all_points + spatial_mean + single_best`** kombinasyonu
  → araştırma bulgusu doğrulandı: **çok-noktalı uzamsal füzyon tek noktayı geçiyor.**
- EPİAŞ realtime-generation notu: `end_date=bugün` 400 verir (tamamlanmamış gün) → **end = bugün-2**
  kullan. Pencere uzunluğu ≥30 gün sorunsuz.
- Kayıt defteri + sonuçlar: `scratchpad/res_registry.json`, `experiment_results.json`.

## EPİAŞ API gerçekleri (doğrulanmış)

- Auth: `POST https://giris.epias.com.tr/cas/v1/tickets` (username/password **body'de**)
  → düz metin `TGT-...` (~2 saat). Veri: header `TGT:` ile
  `https://seffaflik.epias.com.tr/electricity-service/v1`.
- `GET /generation/data/powerplant-list` → yalnızca `id, name, shortName, eic` (1824 santral, 303 RES).
- `POST /generation/data/realtime-generation` `{startDate,endDate,powerPlantId}`, saatlik `items[]`.

## KRİTİK leakage kısıtı

Eğitim verisi `weather_raw WHERE role=previous_runs AND lead_hours=horizon`. Open-Meteo
previous-runs lead'i tam gün katı → **24'ün katı horizon'lar (24, 48, 72…168)** çalışır.
Sub-24h keyfi lead için **Single Runs API** (config-gated, kapalı, ücretli) altyapısı hazır.

## SIRADAKİ İŞ (opsiyonel)

- **Push/merge**: `feature/forecasting-pool` → `main` (kullanıcı GitHub adresi + onay verince).
- ~~**Gerçek santral kurulumu**~~ ✅ TAMAMLANDI (yukarı bak) — 12 RES, 24 champion, hepsi pozitif skill.
- **Ölçekle**: 303 RES'ten daha fazlasını ekle; horizon 72-168h ve solar RES/GES ekle;
  multi-source combo (ecmwf+icon_eu) ve pooled model rakibini gerçek santralde ölç.
- **Serving için forecast-role hava**: reforecast/serving'in gerçek çıktı üretmesi için
  FORECAST-role çok-noktalı hava ingest'i (mevcut ingest previous-runs eğitim verisi çekiyor).
- **Kademeli iyileştirme**: solar plant'larda kPV'yi gerçek veride doğrula; multi-source combo
  ve pooled model rakibini gerçek santralde ölç.

## Önemli endpoint'ler

```
POST /plants                          {plant_id,name,kind,capacity_mw}
POST /plants/{id}/points              {plant_id,point_type,latitude,longitude,...}
POST /plants/{id}/weather-grid        {source,size,upwind_deg?}         → NxN grid
POST /plants/{id}/ingest-weather      {grid_id,sources,past_days,previous_days} | {point_id,...}
POST /plants/{id}/production          (multipart CSV: timestamp,power_mw)
POST /experiments                     {plant_id,point_id,horizons,kind,capacity_mw,point_ids?,source?,...}
GET  /experiments/{id}/champion       kazanan point/source policy + skill
GET  /experiments/{id}/results
GET  /scenario-memory/leaderboard | /scenario-memory/policy-leaderboard
GET  /plants/{id}/scenario-runs | /plants/{id}/strategy-profile
POST /plants/{id}/retrain | /plants/{id}/postprocessor/fit | /plants/{id}/drift/check
POST /plants/{id}/reforecast          {point_id,horizon_hours,kind,issue_time?,w_new?}
GET  /plants/{id}/forecasts?horizon=  | GET /jobs/{id}
```
