# OpenEnergy — Strategy Search ve Geniş Deneme Uzayı

**Tarih:** 2026-06-30  
**Durum:** Hedef tasarım / uygulanacak kapsam  
**Amaç:** OpenEnergy experiment motorunu yalnızca "model seçen" bir AutoML motoru olmaktan çıkarıp, santral ve horizon bazında en iyi **tahmin stratejisini** arayan bir sisteme dönüştürmek.

Bu doküman kullanıcının istediği geniş deneme uzayını tanımlar: feature uzayı, feature-model eşleşmeleri, ensemble stratejileri, dynamic train policy, multi-point weather search, quantile calibration ve AutoGluon/FLAML gibi dış AutoML challenger'ları.

## 1. Ana Fikir

Mevcut sistemde trial şu seviyededir:

```text
PipelineConfig =
  model_family
  feature_blocks
  model_params
  nwp_source
```

Hedef sistemde trial daha geniş bir stratejidir:

```text
StrategyConfig =
  horizon
  data_source_policy
  train_policy
  candidate_pipelines[]
  ensemble_policy
  quantile_policy
  promotion_policy
```

Yani sistem yalnızca şunu aramaz:

```text
LightGBM + wind_power + air_density
```

Bunun yerine şunu arar:

```text
24h horizon için:
  Candidate A: ICON mean/spread + wind_power + direction + LightGBM
  Candidate B: nearest point + gust + shear + CatBoost
  Candidate C: 3-point spatial average + cyclical_time + Ridge
  Train policy: rolling_180d
  Ensemble: validation-weighted p50 average
  Quantile policy: conformal calibration
```

Bu yaklaşımın temel iddiası: enerji tahmininde başarı yalnızca model ailesinden değil, **doğru feature kaynağı + doğru train penceresi + doğru ensemble + doğru horizon ayrımı** kombinasyonundan gelir.

## 2. Arama Eksenleri

| Eksen | Aranacak seçenekler | Not |
|---|---|---|
| Forecast horizon | 24h, 48h; ileride 1h/6h/72h | Previous-runs kısıtı nedeniyle şu an gerçek testte 24/48 öncelikli |
| NWP source | icon, gfs, ecmwf, multi-model mean, spread | Mevcut data path önce çoklu NWP filtrelemeyi desteklemeli |
| Weather point policy | nearest, best_single, average_k, weighted_k, upwind_proxy | Çok noktalı saha optimizasyonu için kritik |
| Feature blocks | wind/solar fiziksel bloklar, temporal, lag, regime, uncertainty | Feature set trial'ın gerçek arama ekseni olmalı |
| Model family | LightGBM, Ridge, RandomForest, XGBoost, CatBoost, Stacking, QuantileGBM | Mevcut genişletilmiş model havuzu |
| AutoML/foundation challenger | AutoGluon Tabular, AutoGluon TimeSeries, Chronos-2/Chronos-Bolt, DirectTabular, FLAML | Ana leakage-safe protokol OpenEnergy'de kalmalı |
| Train policy | expanding, rolling_90/180/365, recent_weighted, seasonal | Dynamic train için ana eksen |
| Ensemble policy | none, simple_mean, weighted_mean, rank_weighted, stacking, QRA | Top-K candidate üstünden denenmeli |
| Quantile policy | native quantile, quantile ensemble, conformal, calibration | p10/p50/p90 güvenilirliği için |
| Promotion policy | best nRMSE, skill-gated, drift-aware challenger | Champion terfisi körü körüne olmamalı |

## 3. Feature Search Uzayı

### 3.1 Rüzgar Feature Grupları

| Grup | Feature'lar | Neden önemli |
|---|---|---|
| Raw wind | wind_speed_10m/80m/100m/120m/180m | Ana sinyal |
| Wind power transform | v, v^2, v^3, clipped/cut-in/cut-out proxy | Türbin güç eğrisi doğrusal değildir |
| Air density | temperature, pressure, humidity, rho, rho_v3 | Aynı rüzgar hızında üretim değişir |
| Direction | sin/cos direction, sector buckets | Wake, arazi ve türbin dizilimi etkisi |
| Shear | speed ratio farklı yükseklikler | Hub-height uyarlaması |
| Gust/turbulence proxy | gust, gust_factor, speed variability | Ramp ve belirsizlik sinyali |
| Temporal | hour/doy/month sin-cos, weekend/season | Mevsimsellik ve günlük pattern |
| NWP uncertainty | model spread, ensemble std, min/max | Belirsizlik ve quantile forecast için |
| Ramp | speed delta, direction delta, forecast ramp | Ani üretim değişimleri |
| Lag/persistence | recent production, rolling mean | Sadece horizon leakage kuralı izin verirse |

### 3.2 Güneş Feature Grupları

| Grup | Feature'lar | Neden önemli |
|---|---|---|
| Irradiance | GHI, DNI, DHI, GTI | Ana PV sinyali |
| Clear-sky | terrestrial radiation, clear_sky_index | Bulut etkisini normalize eder |
| Cloud | low/mid/high/total cloud cover | PV forecast hatasının ana kaynağı |
| Temperature derating | temp_2m, cell_temp proxy, pv_derate | Panel verimi sıcaklıkla düşer |
| Sun geometry | solar elevation/azimuth, daylight flag | Fiziksel üretim penceresi |
| Weather regime | clear/partly_cloudy/cloudy/variable | Her rejimde farklı model daha iyi olabilir |
| Ramp/variability | cloud delta, irradiance ramp | Kısa vadeli belirsizlik |
| Temporal | hour/doy/month sin-cos | Sezonsallık |

### 3.3 Feature Set Candidate Tasarımı

Feature search rastgele tek tek kolon seçimi gibi başlamamalı. Önce anlamlı feature grupları candidate olarak tanımlanmalı:

```python
FeatureSetCandidate(
    name="wind_physical_core",
    blocks=["wind_power", "air_density", "wind_direction"],
)

FeatureSetCandidate(
    name="wind_uncertainty",
    blocks=["wind_power", "wind_gust", "nwp_spread", "cyclical_time"],
)

FeatureSetCandidate(
    name="solar_cloud_regime",
    blocks=["gti", "cloud", "clear_sky_index", "temperature_derating"],
)
```

Sonra Optuna bu feature setleri ve alt bloklarını arayabilir.

## 4. Multi-Point Weather Search

Tek lat/lon çoğu santral için yetersiz kalabilir. Rüzgar sahası genişse veya NWP grid çözünürlüğü kaba ise yakın grid noktalarının kombinasyonu daha iyi sonuç verebilir.

Aranacak point policy seçenekleri:

| Policy | Tanım |
|---|---|
| nearest | Santral koordinatına en yakın weather point |
| best_single | Aday noktalar ayrı ayrı denenir, en iyisi seçilir |
| average_k | En yakın k noktanın ortalaması |
| weighted_k | Mesafe veya validation skoruna göre ağırlıklı ortalama |
| upwind_proxy | Rüzgar yönüne göre yukarı-rüzgar noktasına ağırlık |
| spatial_stack | Her noktanın feature'ı ayrı kolon olarak modele verilir |

İlk uygulanacak basit versiyon:

```text
plant center etrafında 3-5 nokta oluştur
her nokta için weather ingest
point_policy = nearest / best_single / average_3
```

İleri versiyon:

```text
point_policy = weighted_k
weights = Optuna veya validation skorundan öğrenilir
```

## 5. Train Policy Search

Enerji tahmininde "tüm geçmişi kullanmak" her zaman en iyi değildir. Santral davranışı, bakım, kapasite değişimi, çevre etkileri ve NWP model güncellemeleri drift yaratır.

Aranacak train policy seçenekleri:

| Policy | Tanım | Ne zaman iyi olabilir |
|---|---|---|
| expanding_all | Başlangıçtan bugüne tüm veri | Stabil santral, az veri |
| rolling_90d | Son 90 gün | Hızlı drift, mevsimsel yakınlık |
| rolling_180d | Son 180 gün | Pratik default aday |
| rolling_365d | Son 1 yıl | Mevsimsellik + yeterli veri |
| recent_weighted | Eski veri düşük ağırlık | Drift var ama veri atmak istenmiyor |
| seasonal_same_months | Aynı ay/sezon geçmiş yıllar | Güçlü mevsimsellik |
| regime_specific | Hava rejimine göre alt model | Solar cloud regimes / rüzgar sektörleri |

Train policy de trial kaydına yazılmalı. Aksi halde aynı model-feature set farklı train penceresiyle neden iyi/kötü oldu izlenemez.

## 6. Ensemble Search

Kullanıcının özellikle istediği senaryo:

```text
Feature grubu A ile model 1 iyi.
Feature grubu B ile model 2 iyi.
Bunları ensemble yapınca daha iyi olabilir.
```

Bu doğrudan desteklenmeli.

### 6.1 İki Aşamalı Arama

Arama uzayı doğrudan tüm ensemble kombinasyonlarını denerse patlar. Bu yüzden iki aşama:

1. **Single-candidate search**
   - Çok sayıda tekil pipeline dene.
   - Her horizon için top-K aday çıkar.

2. **Ensemble search**
   - Top-K adaylardan kombinasyonlar üret.
   - Simple mean, validation-weighted mean, rank-weighted, stacking, QRA dene.
   - En iyi strategy champion olur.

### 6.2 Ensemble Policy Seçenekleri

| Policy | Tanım | İlk uygulama zorluğu |
|---|---|---|
| none | Tek model | Mevcut davranış |
| simple_mean | p50 ortalaması | Kolay |
| inverse_error_weighted | weight = 1 / validation_error | Kolay |
| rank_weighted | En iyi adaya daha yüksek ağırlık | Kolay |
| optuna_weighted | Ağırlıklar Optuna ile aranır | Orta |
| stacking_meta | Out-of-fold pred üstüne meta model | Zor ama güçlü |
| QRA | Quantile Regression Averaging | Probabilistic forecast için güçlü |

### 6.3 Quantile Ensemble

p10/p50/p90 ayrı ele alınmalı:

```text
p50 = weighted ensemble median prediction
p10/p90 =
  native quantile model varsa kullan
  yoksa residual distribution veya conformal interval ile üret
```

Quantile crossing her zaman düzeltilmeli:

```text
p10 <= p50 <= p90
```

## 7. AutoGluon, TimeSeries ve Foundation Modellerin Yeri

AutoGluon ana beyin olmamalı; çünkü leakage-safe horizon dataset, previous-runs mantığı, dynamic train split ve champion terfisi bizim kontrolümüzde kalmalı. Ama AutoGluon iki ayrı güçlü deney ailesi olarak sisteme girmeli:

1. **AutoGluon Tabular**
   - OpenEnergy'nin hazırladığı horizon-specific feature matrix'i alır.
   - İçeride LightGBM/CatBoost/XGBoost/RF/NN/weighted ensemble gibi modelleri deneyebilir.
   - Custom Strategy'ye karşı AutoML tabular benchmark olur.

2. **AutoGluon TimeSeries / Foundation Models**
   - Target history + known covariates + related series ile çalışır.
   - Chronos-2, Chronos-Bolt, DirectTabular, RecursiveTabular ve desteklenirse TimesFM/TiRex gibi modeller ayrı variant olarak denenir.
   - Day-ahead rolling-origin protokolü yine OpenEnergy tarafından yönetilir.

Doğru kayıt modeli:

```text
engine=autogluon
model_family=tabular_automl
model_variant=best_quality

engine=autogluon
model_family=foundation_timeseries
model_variant=chronos_2

engine=autogluon
model_family=timeseries_tabular
model_variant=direct_tabular
```

Bu sayede AutoGluon şunu yapar:

- verilen leakage-safe feature matrix üzerinde kendi model/ensemble aramasını yapar
- target history + known covariates formatında foundation/time-series modelleri dener
- bizim custom strategy'lerle aynı test setinde ve aynı Scenario Memory leaderboard'da yarışır

Yani AutoGluon "model uzayı"ndan fazlasını sunar ama OpenEnergy'de **fit/predict motoru ve challenger experiment ailesi** olmalıdır.

## 8. Leakage Kuralları

Bu sistem ne kadar genişlerse genişlesin aşağıdaki kurallar değişmez:

1. Tüm split'ler zaman sıralı olmalı; random split yok.
2. Outer test set hiçbir search aşamasında kullanılmamalı.
3. Ensemble ağırlıkları outer test üzerinde öğrenilmemeli.
4. Feature transformları sadece train fold üzerinde fit edilmeli.
5. Production lag feature'ları horizon'a göre sadece tahmin anında bilinebiliyorsa kullanılmalı.
6. Weather feature'ı horizon lead-time ile eşleşmeli: `lead_hours = horizon`.
7. Multi-point selection outer test performansına göre seçilmemeli; inner validation ile seçilmeli.
8. Dynamic train window seçimi outer test'e bakarak yapılmamalı.

## 9. Veri Modeli Genişletme

Mevcut `experiment_trials` tek pipeline için yeterli. Strategy Search için ek tablolar önerilir.

```sql
strategies (
  strategy_id,
  experiment_id,
  plant_id,
  horizon_hours,
  train_policy,
  data_source_policy,
  ensemble_policy,
  quantile_policy,
  cv_nrmse,
  test_nrmse,
  test_pinball,
  skill_score,
  is_champion,
  config_json
)

strategy_candidates (
  candidate_id,
  strategy_id,
  model_family,
  feature_blocks,
  point_policy,
  nwp_source_policy,
  params_json,
  cv_nrmse,
  oof_prediction_path,
  artifact_path
)

strategy_ensemble_members (
  strategy_id,
  candidate_id,
  target,
  weight
)
```

Kısa vadede bu tablolar şart değil; `config_json` ile tek tabloda başlanabilir. Ama kalıcı analiz için normalize şema daha iyi.

## 10. API ve UI

Experiment başlatma isteği genişletilmeli:

```json
{
  "plant_id": "bares-res",
  "point_id": 1,
  "horizons": [24, 48],
  "kind": "wind",
  "capacity_mw": 142.5,
  "n_trials": 100,
  "experiment_type": "openenergy_strategy",
  "engine": "custom",
  "model_family": null,
  "model_variant": null,
  "search_mode": "strategy",
  "search_space": {
    "feature_search": true,
    "ensemble_search": true,
    "train_policy_search": true,
    "multi_point_search": false,
    "automl_challengers": ["flaml", "autogluon"]
  },
  "budget": {
    "max_minutes": 60,
    "top_k_candidates": 12
  }
}
```

AutoGluon Tabular örneği:

```json
{
  "experiment_type": "autogluon_tabular",
  "engine": "autogluon",
  "model_family": "tabular_automl",
  "model_variant": "best_quality",
  "time_limit_seconds": 1800,
  "train_policy": "rolling_180d"
}
```

Chronos-2 örneği:

```json
{
  "experiment_type": "foundation_timeseries",
  "engine": "autogluon",
  "model_family": "foundation_timeseries",
  "model_variant": "chronos_2",
  "prediction_length": 24,
  "known_covariates": ["weather_forecast", "calendar"]
}
```

UI'da basit/ileri seviye ayrımı olmalı:

- Basit mod: "Hızlı / Dengeli / Derin" preset.
- İleri mod: feature search, ensemble search, train policy, multi-point, AutoML challengers checkbox'ları.

## 11. Uygulama Sırası

### Faz 1 — Strategy Config Temeli

- `PipelineConfig` yanına `CandidateConfig`, `StrategyConfig`, `TrainPolicyConfig` ekle.
- Mevcut tek model akışını strategy'nin `ensemble_policy=none` hali olarak çalıştır.
- Trial kayıtlarında `strategy_config_json` sakla.

### Faz 2 — Train Policy Search

- `expanding_all`, `rolling_90d`, `rolling_180d`, `rolling_365d` ekle.
- Split fonksiyonları train policy'ye göre dataset'i kırpsın.
- Her trial train policy bilgisini kaydetsin.

### Faz 3 — Top-K Candidate Arama

- Tekil pipeline'ları çalıştır.
- Her horizon için top-K candidate sakla.
- OOF/validation prediction üret.

### Faz 4 — Ensemble Search

- Top-K adaylardan simple mean ve inverse-error weighted ensemble ekle.
- Sonra Optuna-weighted ensemble ekle.
- En iyi ensemble strategy champion olabilir.

### Faz 5 — Quantile Calibration

- Native quantile modelleri kullan.
- Native quantile yoksa residual/conformal interval üret.
- Quantile monotonicity düzelt.
- Pinball ve calibration metriklerini strategy selection'a ekle.

### Faz 6 — Multi-Point Weather Search

- Santral çevresinde aday weather point üretimi.
- `best_single`, `average_k`, `weighted_k` policy'leri.
- Point policy'yi feature matrix assembly içine taşı.

### Faz 7 — AutoML / Foundation Challenger

- AutoGluon Tabular wrapper ekle.
- AutoGluon TimeSeries dataset adapter ekle.
- Chronos-2, Chronos-Bolt, DirectTabular, RecursiveTabular variant'larını ayrı seçenek olarak aç.
- Time budget, prediction length ve known covariates parametreleri ver.
- Ana leakage-safe split/rolling-origin motoru OpenEnergy'de kalsın.

## 12. Presetler

| Preset | Amaç | İçerik |
|---|---|---|
| quick | 5-10 dk hızlı deneme | az model, az feature set, ensemble yok |
| balanced | günlük kullanım | feature search + model search + simple ensemble |
| deep | en iyi sonuç arama | top-K ensemble + train policy + quantile calibration |
| research | geniş keşif | multi-point + AutoML challenger + Optuna weights |

## 13. Başarı Metrikleri

Her horizon için:

- nRMSE
- nMAE
- bias
- skill score vs persistence/climatology
- pinball loss
- quantile coverage
- interval width
- calibration error
- runtime
- model artifact size

Champion seçimi yalnız nRMSE'ye bakmamalı. Minimum kural:

```text
champion = lowest nRMSE
  AND skill_score > 0
  AND no severe quantile calibration failure
```

İleride multi-objective selection:

```text
score = nRMSE + lambda_pinball * pinball + lambda_runtime * runtime_penalty
```

## 14. Literatürden Çıkan Tasarım İlkeleri

Bu tasarımın arkasındaki güçlü literatür sinyalleri:

- Weather ensemble ve model-spread feature'ları rüzgar/güneş tahmininde güçlü sinyal taşır.  
  Referans: [Probabilistic Wind Power Forecasting with Tree-Based Machine Learning and Weather Ensembles](https://arxiv.org/abs/2602.13010)
- Continuous learning / bias correction gerçek santral forecast'lerinde ciddi iyileşme sağlar.  
  Referans: [Bias correction of wind power forecasts with SCADA data and continuous learning](https://arxiv.org/abs/2402.13916)
- PV tarafında ensemble forecast post-processing ve quantile regression kalibrasyonu önemlidir.  
  Referans: [Post-processing of ensemble photovoltaic power forecasts with distributional and quantile regression methods](https://arxiv.org/abs/2508.15508)
- NWP ensemble post-processing solar irradiance tahminlerinde bias ve calibration hatasını azaltır.  
  Referans: [Post-processing numerical weather prediction ensembles for probabilistic solar irradiance forecasting](https://arxiv.org/abs/2101.06717)
- Spatial context ve çoklu nokta kullanımı multi-horizon wind forecast için güçlüdür.  
  Referans: [A Spatiotemporal Deep Neural Network for Fine-Grained Multi-Horizon Wind Prediction](https://arxiv.org/abs/2309.04733)
- Güneş tarafında clear-sky index faydalı ama tek doğru yaklaşım değildir; clear-sky kullanan ve kullanmayan feature aileleri birlikte aranmalıdır.  
  Referans: [On the Importance of Clearsky Model in Short-Term Solar Radiation Forecasting](https://arxiv.org/abs/2503.07647)

## 15. Nihai Hedef

OpenEnergy'nin nihai experiment motoru şu soruya cevap vermeli:

```text
Bu santral ve bu horizon için:
  hangi hava kaynağı,
  hangi nokta/kombinasyon,
  hangi feature grubu,
  hangi model veya model grubu,
  hangi train penceresi,
  hangi ensemble ve quantile calibration
en iyi, güvenilir ve üretime alınabilir tahmini verir?
```

Bu hedefe ulaşıldığında sistem klasik bir AutoML aracı değil, santral bazlı **forecast strategy discovery engine** olur.
