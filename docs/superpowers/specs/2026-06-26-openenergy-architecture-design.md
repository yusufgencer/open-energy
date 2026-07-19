# OpenEnergy — Platform Mimarisi ve Yol Haritası

**Tarih:** 2026-06-26
**Durum:** Onaylandı (mimari + yol haritası)
**Kapsam:** Bu doküman tüm platformun hedef mimarisini ve alt-projelere bölünmüş yol haritasını tanımlar. Her alt-proje kendi detaylı spec → plan → implementation döngüsünü ayrıca alacaktır.

---

## 1. Vizyon

Rüzgâr ve güneş enerjisi için **piyasanın en doğru üretim tahminini** üreten bir tahmin platformu (enercast benzeri). Kullanıcı kendi santralinin geçmiş üretim verisini ekler, lokasyon ve nokta(lar)ını işaretler; sistem o santral için **en iyi tahmin algoritmasını otomatik bulur** (experiment/AutoML), forecast horizon'a göre, **kesinlikle veri sızıntısı (data leakage) olmadan**. Modeller seçili feature-engineering kombinasyonuyla düzenli olarak yeniden eğitilebilir (retrain).

## 2. Temel Kararlar (brainstorming çıktısı)

| Karar | Seçim |
|---|---|
| Teknoloji yığını | Python + FastAPI + DuckDB + Next.js |
| Ölçek/dağıtım | Lokal / tek kullanıcı, kendi santraller (auth/multi-tenant YOK; veri modeli ileride SaaS'a taşınabilir tutulur) |
| Üretim tipi | Rüzgâr **ve** güneş (mimari ikisini de baştan destekler) |
| İlk hava sağlayıcı | Open-Meteo (tam destek; 4 API) — provider abstraction ile yenileri eklenebilir |
| Optimizasyon hedefi | Horizon başına **nRMSE** (RMSE / kurulu kapasite) minimizasyonu (default; nMAE/pinball'a konfigüre edilebilir) |
| Kaynak kontrolü | **GitLab'a ASLA push edilmez.** Lokal commit serbest. GitHub remote'a push yalnız kullanıcı URL verince ve onaylayınca. |

## 3. Repo / Bileşen Mimarisi

```
~/Projects/openenergy/
├── backend/                      # Python, FastAPI
│   ├── openenergy/
│   │   ├── providers/            # Hava verisi sağlayıcı abstraction
│   │   │   ├── base.py           #   WeatherProvider arayüzü (ABC)
│   │   │   └── openmeteo/        #   forecast, archive, previous_runs, ensemble
│   │   ├── assets/               # Santral/varlık modeli: plant, points, capacity, specs
│   │   ├── ingestion/            # Veri çekme + production CSV yükleme + hizalama
│   │   ├── features/             # Feature engineering (rüzgâr & güneş transformları)
│   │   ├── experiments/          # ⭐ Experiment/AutoML motoru (kalbi)
│   │   ├── models/               # Model wrapper'ları (LightGBM/XGBoost/linear...)
│   │   ├── evaluation/           # Leakage-safe CV, metrikler, skill score
│   │   ├── forecasting/          # Canlı forecast üretimi (champion model)
│   │   ├── retraining/           # Scheduler + retrain job'ları
│   │   ├── storage/              # DuckDB/Parquet erişim katmanı
│   │   └── api/                  # FastAPI route'ları
│   └── jobs/                     # Uzun süren işler (experiment, retrain) — arka plan runner
├── frontend/                     # Next.js + harita (MapLibre/Leaflet)
├── data/                         # DuckDB dosyası + Parquet (gitignore)
└── docs/superpowers/specs/       # Spec dokümanları
```

**Tasarım ilkeleri:**
- **Provider abstraction:** `WeatherProvider` ABC → `OpenMeteoProvider` ilk implementasyon. Open-Meteo'nun 4 API'si tek provider içinde rollerine göre ayrılır (geçmiş/önceki-run/forecast/ensemble). Yeni provider = yeni sınıf.
- **Uzun süren işler** (experiment, retrain) FastAPI request'ini bloklamaz. Lokal olduğu için ağır kuyruk (Celery/Redis) yerine **in-process background job runner + DuckDB `jobs` durum tablosu** ile başlanır. İleride worker'a terfi edilebilir (sınır net).
- **Storage:** DuckDB tek dosya + büyük zaman serileri için Parquet.
- **İzole, tek-sorumluluklu modüller:** her modül net arayüzle iletişir, bağımsız test edilebilir.

## 4. Veri Modeli (DuckDB tabloları)

| Tablo | İçerik |
|---|---|
| `plants` | id, ad, tip (wind/solar), kurulu kapasite (MW), zaman dilimi |
| `plant_points` | plant'e ait coğrafi noktalar: tip (`weather`/`turbine`/`panel_array`), lat/lon, hub yüksekliği veya tilt/azimuth |
| `production` | yüklenen gerçek üretim: plant_id, timestamp, power_mw (gözlenen hedef) |
| `weather_raw` | provider'dan çekilen ham hava: point_id, kaynak API, NWP modeli, valid_time, issue_time/lead, değişkenler |
| `features` | işlenmiş feature matrisi (horizon'a göre materialize) |
| `experiments` | deney koşusu: config, arama uzayı, durum, en iyi sonuç referansı |
| `experiment_trials` | her trial: feature seti + model + hyperparam + NWP kaynağı + per-horizon skor |
| `models` | eğitilmiş model artefaktları (champion dahil), metadata, geçerlilik aralığı |
| `forecasts` | üretilen tahmin: plant_id, issue_time, valid_time, horizon, p10/p50/p90 |
| `jobs` | arka plan iş durumu (experiment/retrain), ilerleme, log |

**Leakage sınırı (kritik):** Her feature satırı `(valid_time, issue_time, horizon)` üçlüsü taşır. Bir horizon datasetinde hava feature'ı, o horizon'a karşılık gelen **lead-time forecastinden** gelir (ör. day-ahead için Open-Meteo `_previous_day1`), gözlenen üretimle eşlenir. ERA5/Archive yalnızca fiziksel eşlemeyi öğrenmek ve backfill için kullanılır; final değerlendirme her zaman forecast-kaliteli girdiyle yapılır.

## 5. Veri Akışı (uçtan uca)

1. **Kurulum:** Santral oluştur → haritada nokta(lar) işaretle (weather/turbine/panel) → üretim CSV yükle (timestamp, power).
2. **Backfill:** Archive API (ERA5) ile geçmiş hava + Previous Runs API (`_previous_dayN`) ile "tahminin o an ne dediği" → production ile timestamp hizalama.
3. **Feature:** Rüzgâr (v, v², v³, ρ-düzeltme, yön sin/cos, shear, model-spread) ve güneş (GTI, clear-sky index, bulut, sıcaklık derating) + cyclical zaman → her forecast horizon için ayrı materialize.
4. **Experiment:** Arama uzayında yüzlerce trial → her biri walk-forward + embargolu CV ile değerlendirilir → horizon başına en iyi pipeline (champion) seçilir.
5. **Forecast:** Champion model + canlı Forecast/Ensemble API → p10/p50/p90 tahmin.
6. **Retrain:** Scheduler (ör. günlük) → seçili feature+model kombinasyonuyla sliding/expanding window retrain → champion/challenger → kazanan terfi.

## 6. Experiment / AutoML Motoru (kalbi)

### 6.1 Arama uzayı

| Eksen | Seçenekler |
|---|---|
| NWP kaynağı | `icon`, `gfs`, `ecmwf`, çoklu-model stack, ensemble-mean, model-spread feature'lı |
| Feature seti | açılıp-kapanabilir bloklar: ham rüzgâr, cubed+ρ-düzeltme, güç-eğrisi transformu, yön sin/cos, shear, clear-sky index, GTI, bulut katmanları, cyclical zaman, lag'ler (yalnız izin verilen horizon'da) |
| Model ailesi | LightGBM, XGBoost, CatBoost, regularized linear (MOS), random forest, (opsiyonel) stacking |
| Hyperparametre | Optuna (Bayesian TPE) ile model başına |
| Hedef tipi | deterministik (p50) + quantile (p10/p90, pinball loss) |

**Her forecast horizon için AYRI çalışır** (ör. 1s, 6s, 24s, 38s, gün-öncesi kapı). Kısa horizon lag/persistence kullanabilir; uzun horizon saf NWP + model-spread'e dayanır.

### 6.2 Leakage-safe değerlendirme (iki katmanlı — pazarlık konusu değil)

```
Dış katman: Walk-forward zaman bölme — train[başlangıç…T] / test[T…T+Δ], ileri kaydır.
            Dış test setine arama ASLA dokunmaz → dürüst final skor.

İç katman (her trial): Optuna objektifi = blocked + embargolu CV
  - Feature transformları yalnız train fold'unda fit edilir.
  - Embargo ≥ max(rolling pencere, horizon) → autokorelasyon sızıntısı yok.
  - Hava feature'ı = o horizon'un lead-time forecasti (_previous_dayN), gözlenen güçle eşli.

Baseline: her horizon için persistence + climatology → SKILL SCORE raporlanır.
```

### 6.3 Seçim kriteri ve metrikler

- **Birincil hedef (default):** dokunulmamış walk-forward test setinde en düşük **nRMSE** (RMSE / kurulu kapasite), horizon başına.
- **Konfigüre edilebilir:** hedef nMAE veya pinball loss'a çevrilebilir.
- **Güvenlik bekçisi:** nRMSE en düşük olsa bile model persistence/climatology baseline'ını **skill score** ile yenmiyorsa kırmızı bayrak (raporlanır, körü körüne terfi yok).
- **Her zaman raporlanan yan metrikler:** nMAE, bias (nMBE), pinball loss, skill score, reliability.

### 6.4 Çıktı ve ölçek

- Çıktı: horizon başına **champion pipeline** = {NWP kaynağı + feature seti + model + hyperparam} + dürüst per-horizon metrikler. `experiment_trials` ve `models` tablolarına yazılır; en iyisi forecasting'e terfi eder.
- Ölçek: "yüzlerce senaryo" tam grid değil — Optuna TPE + trial/süre bütçesi. Paralel trial ≈ CPU çekirdek − 2. Kullanıcı experiment başlatırken bütçe ve horizon listesi seçer.

## 7. Open-Meteo Entegrasyonu (sağlayıcı notları)

Open-Meteo'nun 4 API'si rollerine göre kullanılır:

| API | Endpoint | Rol |
|---|---|---|
| Forecast | `api.open-meteo.com/v1/forecast` | Canlı operasyonel tahmin |
| Historical/Archive | `archive-api.open-meteo.com/v1/archive` | ERA5 reanalysis — geçmiş backfill, fiziksel eşleme |
| Previous Runs | `previous-runs-api.open-meteo.com/v1/forecast` | `_previous_dayN` — leakage-safe eğitim (tahminin o an dediği) |
| Ensemble | `ensemble-api.open-meteo.com/v1/ensemble` | Olasılıksal/belirsizlik girdileri |

**Kritik değişkenler:**
- Rüzgâr: `wind_speed_80m/120m/180m`, `wind_direction_*`, `wind_gusts_10m`, `temperature_2m`, `surface_pressure` (hava yoğunluğu için), `relative_humidity_2m`.
- Güneş: `global_tilted_irradiance` (GTI; `tilt`/`azimuth` ile), `shortwave_radiation` (GHI), `direct_normal_irradiance` (DNI), `diffuse_radiation` (DHI), `terrestrial_solar_radiation` (clear-sky index için), `cloud_cover*`, `temperature_2m`, `is_day`.
- Çoklu model: `models=icon,gfs,ecmwf` → model-arası spread güçlü belirsizlik feature'ı.

**Pratik kısıtlar (build öncesi doğrulanacak):** Previous Runs API çoğu modelde ~Oca 2024'ten itibaren; free tier ~10k çağrı/gün ve throttle. Ticari kullanım API key + `customer-` host gerektirir. Ingestion batch + cache + 429 backoff ile tasarlanır.

## 8. Yol Haritası — Alt-projeler

Her alt-proje kendi spec → plan → implementation döngüsünü alır:

| # | Alt-proje | Kapsam | Bağımlılık |
|---|---|---|---|
| 1 | **Veri temeli + provider** | DuckDB şeması, `WeatherProvider` ABC, `OpenMeteoProvider` (4 API), production CSV yükleme, timestamp hizalama, leakage-safe horizon dataset üretici | — |
| 2 | **Feature engineering** | Rüzgâr + güneş transform blokları, açılıp-kapanabilir feature setleri, materialize | 1 |
| 3 | **Experiment/AutoML motoru** | Arama uzayı, walk-forward + embargolu CV, Optuna, per-horizon champion seçimi, nRMSE hedefi, metrik raporu | 1, 2 |
| 4 | **Forecast serving + retrain** | Champion ile canlı forecast (p10/p50/p90), scheduler, champion/challenger retrain | 3 |
| 5 | **Frontend** | Santral yönetimi, haritada nokta işaretleme, CSV yükleme UI, experiment başlatma + sonuç dashboard, doğruluk raporları | 1–4 |

**Sıradaki adım:** Alt-proje 1 (Veri temeli + provider) için detaylı spec.

## 9. Açık Sorular (Alt-proje 1 spec'inde netleşecek)

- Gerçek bir test santralinin geçmiş üretim verisi var mı, ne kadar geçmiş? (model doğrulama için kritik)
- Open-Meteo free tier yeterli mi yoksa ticari key mi gerekecek?
- Production CSV'nin tam formatı (zaman dilimi, granülarite: saatlik/15 dk, kolon adları)?
- Hedef forecast horizon listesi (varsayılan: 1s, 6s, 24s, 38s, gün-öncesi kapı)?
