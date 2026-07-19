# OpenEnergy — Rüzgar/Solar Araştırma Bulguları ve Senaryo Hafızası

**Tarih:** 2026-06-30  
**Durum:** Araştırma sentezi + ürünleştirilecek tasarım  
**Amaç:** Rüzgar ve solar üretim tahmini için literatürde gerçekten başarı sağlamış feature engineering, ensemble, dynamic training ve probabilistic post-processing yöntemlerini OpenEnergy'nin Strategy Search motoruna çevirmek.

Bu dokümanın hedefi yalnızca "hangi makale iyi?" listesini tutmak değil. Asıl hedef:

```text
Santral bazında hangi senaryolar tutuyor?
Hangi feature/model/train/ensemble kombinasyonları tekrar tekrar kazanıyor?
Yeni santral geldiğinde geçmiş başarı hafızasından nasıl daha iyi başlarız?
```

## 1. Ürün İlkesi

OpenEnergy'de nihai arama problemi şöyle tanımlanmalı:

```text
plant_id + horizon için:
  weather source/policy
  weather point policy
  feature family
  model family
  train policy
  ensemble policy
  quantile calibration
  promotion policy
kombinasyonlarından hangisi en iyi forecast stratejisini üretir?
```

Bu yüzden sistemin öğrenmesi gereken şey yalnız tekil model değil, **senaryo başarısıdır**.

## 2. Rüzgar İçin En Güçlü Bulgular

Rüzgar tarafında araştırma sentezinin kısa sonucu:

```text
En güçlü ilk strateji =
  NWP ensemble mean/spread/quantile
  + multi-point spatial aggregate
  + rolling/continuous retrain
  + production-level quantile calibration
```

Model mimarisinden önce korunması gerekenler: issue-time güvenli backtest, curtailment/bakım/veri kalite bayrakları, hub-height uyarlaması, kalibrasyon metrikleri.

### W1 — NWP Ensemble ve Spread Feature'ları Çok Değerli

Kaynak: [Probabilistic Wind Power Forecasting with Tree-Based Machine Learning and Weather Ensembles](https://arxiv.org/abs/2602.13010)

Bulgu:
- Tree-based ML + weather ensemble day-ahead wind power forecast'te güçlü.
- Weather ensemble kullanımı point accuracy'yi ciddi artırabiliyor.
- Probabilistic yöntemlerde conformalized quantile regression, NGBoost ve diffusion tabanlı yaklaşımlar karşılaştırılıyor.

OpenEnergy senaryosu:

```text
wind_nwp_ensemble_core =
  features:
    wind_speed_mean
    wind_speed_std/spread
    wind_direction_mean
    pressure/temp
    model disagreement
  models:
    LightGBM, CatBoost, XGBoost, QuantileGBM
  quantile:
    native quantile + conformal calibration
```

Öncelik: Çok yüksek.  
Risk: Mevcut ingestion/data path önce çoklu NWP kaynağı ve model bazlı pivot desteklemeli.

### W2 — Continuous Learning / Dynamic Train Baseline Değil, Ana Eksendir

Kaynak: [Bias correction of wind power forecasts with SCADA data and continuous learning](https://arxiv.org/abs/2402.13916)

Bulgu:
- 48h NWP forecast bias correction, ham NWP baseline'a göre büyük iyileşme sağlıyor.
- Çalışma, mimari değişikliklerden çok pipeline değişikliklerinin daha önemli olabileceğini vurguluyor.
- Continuous learning stratejisi yeni veri geldikçe performansı artırıyor.

OpenEnergy senaryosu:

```text
wind_continuous_bias_correction =
  train_policy:
    rolling_90d
    rolling_180d
    expanding_all
    recent_weighted
  target:
    production residual veya normalized production
  promotion:
    challenger sadece incumbent'ı geçerse terfi eder
```

Öncelik: Çok yüksek.  
Risk: Veri azsa rolling window overfit edebilir; minimum sample guard gerekir.

### W3 — Spatial Context / Multi-Point Weather Search Operasyonel Değer Taşır

Kaynaklar:
- [A Spatiotemporal Deep Neural Network for Fine-Grained Multi-Horizon Wind Prediction](https://arxiv.org/abs/2309.04733)
- [Concurrent CNN-RNN for Multi-Step Wind Power Forecasting](https://arxiv.org/abs/2301.00819)

Bulgu:
- NWP, lokal gözlem, spatial correlation, covariate selection ve ensemble modülü birlikte kullanılıyor.
- Operasyonel platforma entegre edilmiş olması önemli.
- Birden çok NWP modeli, rüzgar çiftliği ve coğrafi grid noktasından spatial-temporal bilgi çıkarmak bazı durumlarda tekil santral eğitiminden daha iyi.

OpenEnergy senaryosu:

```text
wind_spatial_point_search =
  point_policy:
    nearest
    best_single
    average_3
    weighted_3
    spatial_stack
  feature_policy:
    her point için wind_speed/direction ayrı kolon
    veya point mean/spread
```

Öncelik: Çok yüksek.  
Risk: Weather ingestion maliyeti artar; batch yazım ve cache şart.

MVP notu:

```text
Deep CNN/RNN'e atlamadan önce:
  3x3/5x5 grid patch'ten tree-model feature'ları üret
  mean/std/min/max
  upwind/downwind özetleri
  spatial gradient
```

Bu daha hızlı, daha açıklanabilir ve mevcut LightGBM/CatBoost/XGBoost havuzuyla uyumlu.

### W4 — Multi-Scale Temporal / Horizon-Specific Feature Set Gerekli

Kaynak: [Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer](https://arxiv.org/abs/2305.18724)

Bulgu:
- Uzun horizon için spatial-temporal representation ve farklı temporal ölçekler önemli.
- Tek feature set tüm horizon'lara aynı şekilde uygulanmamalı.

OpenEnergy senaryosu:

```text
horizon_specific_wind_strategy =
  24h:
    NWP + gust + direction + recent bias
  48h:
    NWP ensemble spread + temporal seasonality + spatial mean
```

Öncelik: Yüksek.  
Risk: Horizon başına ayrı champion yönetimi zaten var; Strategy Search ile genişletilmeli.

### W5 — Missingness / Robust Quantile Modeling Pratikte Önemli

Kaynak: [Probabilistic wind power forecasting resilient to missing values](https://arxiv.org/abs/2305.14662)

Bulgu:
- Eksik sensör/weather değerleri probabilistic forecast'i bozabilir.
- Missingness pattern'ını modelin parçası yapmak iyi sonuç verebilir.

OpenEnergy senaryosu:

```text
missingness_features =
  is_missing_wind_speed
  is_missing_direction
  n_missing_weather_vars
  fallback_source_used
```

Öncelik: Orta-yüksek.  
Risk: İlk fazda basit imputation + missing flags yeterli.

### W6 — Üretim Seviyesinde Ensemble Post-Processing Weather Seviyesinden Daha Kritik Olabilir

Kaynak: [Evaluating Ensemble Post-Processing for Wind Power Forecasts](https://arxiv.org/abs/2009.14127)

Bulgu:
- NWP ensemble ham halde biaslı/underdispersed olabilir.
- Sadece weather forecast'i düzeltmek yeterli olmayabilir.
- Final power forecast seviyesinde calibration/sharpness iyileştirmesi daha doğrudan değer üretir.

OpenEnergy senaryosu:

```text
wind_power_level_calibration =
  base_models:
    LightGBM
    CatBoost
    XGBoost
    QuantileGBM
  postprocess:
    conformal intervals
    residual quantile correction
    QRA over top-K candidates
  metrics:
    pinball
    coverage
    interval_width
    reliability
```

Öncelik: Çok yüksek.  
Risk: nRMSE tek başına yeterli olmaz; calibration metrikleri leaderboard'a girmeli.

### W7 — Hub-Height Wind Speed Kalibrasyonu Ayrı Bir Alt Problem

Kaynak: [Calibration of wind speed ensemble forecasts for power generation](https://arxiv.org/abs/2104.14910)

Bulgu:
- Hub-height wind speed ensemble forecast'ları ham halde biaslı olabilir.
- EMOS/MOS benzeri post-processing kalibrasyonu iyileştirir.

OpenEnergy senaryosu:

```text
hub_height_mos =
  weather_target:
    wind_speed_100m/120m calibrated proxy
  features:
    ensemble mean/std/skew
    direction sector
    temp/pressure
  second_stage:
    calibrated wind -> production model
```

Öncelik: Orta-yüksek.  
Risk: Hub-height ve türbin teknik bilgisi yoksa proxy kalitesi sınırlı kalır.

### W8 — Deep/Graph/Transformer Modeller Araştırma Adayı, Ama Tabular Baseline'ı Geçme Şartıyla

Kaynaklar:
- [Hierarchical Spatial-Temporal Transformer Network for Wind Power Forecasting](https://arxiv.org/abs/2305.18724)
- [Systematic Evaluation of Current Architectures in Wind Power Forecasting](https://arxiv.org/abs/2606.02849)
- [GenCast: Diffusion-based ensemble weather forecasting](https://arxiv.org/abs/2312.15796)

Bulgu:
- Spatial-temporal transformer, hybrid ve decomposition tabanlı yaklaşımlar umut verici.
- Ancak gerçek dünya validasyonu ve coverage/width dengesi hâlâ zor.
- ML weather ensemble kaynakları ileride NWP source olarak çok değerli olabilir.

OpenEnergy senaryosu:

```text
research_deep_wind =
  only if:
    tabular strategy benchmark established
    Scenario Memory shows plateau
  candidate:
    graph/spatial transformer
    CNN-RNN over NWP patch
    ML weather source features
```

Öncelik: P2.  
Risk: Veri miktarı, bakım maliyeti ve açıklanabilirlik.

## 3. Solar İçin En Güçlü Bulgular

Solar tarafında araştırma sentezinin kısa sonucu:

```text
En güçlü ilk strateji =
  clear-sky / solar geometry / cloud-regime features
  + NWP ensemble mean/spread/quantiles
  + fiziksel PV model-chain baseline
  + PV power seviyesinde quantile calibration
  + rolling/adaptive training
```

Solar için model hatalarının büyük kısmı model ailesinden değil, cloud regime, clear-sky/irradiance dönüşümü, panel metadata ve drift/curtailment ayrımından gelir.

### S1 — PV Forecast'te Post-Processing ve Quantile Regression Kritik

Kaynaklar:
- [Post-processing of ensemble photovoltaic power forecasts with distributional and quantile regression methods](https://arxiv.org/abs/2508.15508)
- [Probabilistic Photovoltaic Power Forecasts: Which Post-Processing Stage Is Best?](https://arxiv.org/abs/2406.04424)

Bulgu:
- Ham PV ensemble forecast'ler biaslı ve kalibrasyonsuz olabiliyor.
- İstatistiksel post-processing tüm ham ensemble'a göre iyileştirme sağlıyor.
- Non-parametric ve ML tabanlı quantile regression yöntemleri güçlü.
- NWP -> GHI -> PV model-chain içinde pratikte çoğu zaman final PV power çıktısını post-process etmek en güçlü/kolay yol.

OpenEnergy senaryosu:

```text
solar_quantile_postprocess =
  base:
    PV p50 models
    raw ensemble features
  calibrator:
    quantile regression averaging
    conformal intervals
    isotonic/monotonic correction
```

Öncelik: Çok yüksek.  
Risk: p10/p90 kalibrasyonu için yeterli validation dönemi gerekir.

MVP notu:

```text
İlk fazda GHI kalibrasyonuna değil,
modelin ürettiği PV power forecast'inin tesis+horizon bazlı kalibrasyonuna odaklan.
```

### S2 — Solar Irradiance İçin NWP Ensemble Post-Processing 48h'ye Kadar Güçlü

Kaynak: [Post-processing numerical weather prediction ensembles for probabilistic solar irradiance forecasting](https://arxiv.org/abs/2101.06717)

Bulgu:
- Ensemble NWP irradiance forecast'leri sistematik bias ve calibration hatası taşıyor.
- Post-processing, lead time 48h ve ötesinde tutarlı iyileşme sağlıyor.

OpenEnergy senaryosu:

```text
solar_irradiance_nwp_postprocess =
  features:
    GHI/DNI/DHI/GTI ensemble mean
    ensemble spread
    cloud cover spread
  model:
    QuantileGBM
    QRA
    conformal calibration
```

Öncelik: Çok yüksek.  
Risk: Open-Meteo'dan ensemble/previous-runs değişkenlerinin tutarlı alınması gerekir.

### S3 — Weather Regime Bazlı Feature Selection PV'de Başarı Sağlıyor

Kaynak: [Feature Construction and Selection for PV Solar Power Modeling](https://arxiv.org/abs/2202.06226)

Bulgu:
- Weather type bazlı feature selection klasik SVM/RF/GBDT baseline'larını geçiyor.
- Feature construction ve selection, model ailesi kadar önemli.

OpenEnergy senaryosu:

```text
solar_regime_strategy =
  regimes:
    clear
    partly_cloudy
    cloudy
    high_variability
  per_regime:
    feature set + model + calibration ayrı seçilir
```

Öncelik: Yüksek.  
Risk: Regime sınıfları doğru tasarlanmazsa veri parçalanır.

### S4 — Clear-Sky Index Faydalı ama Mutlak Doğru Değil

Kaynak: [On the Importance of Clearsky Model in Short-Term Solar Radiation Forecasting](https://arxiv.org/abs/2503.07647)

Bulgu:
- Clear-sky index yaygın ve güçlü ama operational modellerde clear-sky bağımlılığı her zaman ideal olmayabilir.
- Clear-sky kullanan ve kullanmayan feature aileleri birlikte denenmeli.

OpenEnergy senaryosu:

```text
solar_clear_sky_ablation =
  Candidate A:
    raw irradiance + cloud + temp
  Candidate B:
    clear_sky_index + GTI + cloud
  Candidate C:
    sun geometry + raw weather
  ensemble:
    regime-weighted
```

Öncelik: Yüksek.  
Risk: Güneş geometrisi hesapları ve panel tilt/azimuth metadata kalitesi önemli.

### S5 — Probabilistic / Scenario Forecasting Sadece Güven Bandı Değil, Operasyonel Değer

Kaynak: [A deep generative model for probabilistic energy forecasting in power systems: normalizing flows](https://arxiv.org/abs/2106.09370)

Bulgu:
- Wind, solar ve load için weather-based scenario forecasting ele alınıyor.
- Değer değerlendirmesi yalnız hata metriği değil, enerji retailer operasyonu gibi karar değerleriyle de yapılabiliyor.

OpenEnergy senaryosu:

```text
scenario_quality_metrics =
  pinball_loss
  coverage
  interval_width
  forecast_value_proxy
```

Öncelik: Orta-yüksek.  
Risk: İlk ürün için p10/p50/p90 yeterli; tam senaryo üretimi sonra.

### S6 — Fiziksel PV Model-Chain ve pvlib Baseline Mutlaka Olmalı

Kaynaklar:
- [Clear-Sky Index-Based GHI Forecasting and PVLib model chain](https://arxiv.org/abs/2508.07462)
- [pvlib python: a python package for modeling solar energy systems](https://doi.org/10.21105/joss.00884)

Bulgu:
- Clear-sky irradiance, actual/cloudy irradiance, cloud type ve meteorolojik değişkenlerle fiziksel PV output zinciri kurulabiliyor.
- pvlib, PV sistem modellemesi için standart açık kaynak araçlardan biri.

OpenEnergy senaryosu:

```text
solar_physical_model_chain =
  inputs:
    capacity_mw
    tilt
    azimuth
    latitude/longitude
    temperature
    wind
    irradiance
  outputs:
    physical_p50_baseline
    clipping_indicator
    module_temperature_proxy
  ML:
    residual correction over physical baseline
```

Öncelik: Çok yüksek.  
Risk: Tilt/azimuth/inverter metadata yoksa baseline biaslı olur; ama residual model bunu kısmen düzeltebilir.

### S7 — Adaptive Training / Concept Drift Solar'da Ana Problem

Kaynak: [Adaptive LSTM for Concept Drift in PV Forecasting](https://arxiv.org/abs/2109.13442)

Bulgu:
- PV kapasite değişimi, inverter davranışı, arıza, bakım, kirlenme ve curtailment offline modelleri bozar.
- Yeni veriden dinamik öğrenme day-ahead PV forecast'i iyileştirebilir.

OpenEnergy senaryosu:

```text
solar_adaptive_training =
  train_policy:
    rolling_30d
    rolling_90d
    rolling_180d
    recent_residual_correction
  drift:
    residual mean shift
    daytime-only degradation
    capacity/clipping change
```

Öncelik: Çok yüksek.  
Risk: Anomaliyi normal davranış sanmamak için bakım/curtailment/veri kalite flag'leri gerekir.

### S8 — Komşu Tesis / Spatial Residual Feature'ları Portföyde Çok Değerli

Kaynak: [Spatio-temporal graph neural networks for multi-site PV forecasting](https://arxiv.org/abs/2107.13875)

Bulgu:
- Çok sayıda PV sistemi sanal hava istasyonu gibi kullanılabilir.
- Spatial-temporal ilişkiler özellikle 0-6h horizonlarda güçlüdür.

OpenEnergy senaryosu:

```text
solar_neighbor_residuals =
  if portfolio available:
    distance_weighted_neighbor_clear_sky_deviation
    neighbor_recent_residual
    regional_cloud_shock
  first model:
    tabular spatial residual features
  later:
    graph model
```

Öncelik: Portföy varsa yüksek, tek tesis MVP'sinde orta.  
Risk: Zaman senkronizasyonu ve komşu tesis metadata gerekir.

### S9 — Intraday Solar İçin Uydu/Kamera Feature'ları Ayrı Roadmap

Kaynaklar:
- [SolarSTEPS: Probabilistic Cloud Nowcasting](https://arxiv.org/abs/2305.07293)
- [SkyGPT: Probabilistic Short-term PV Forecasting with Sky Images](https://arxiv.org/abs/2306.11682)
- [Data-constrained short-term irradiance forecasting](https://arxiv.org/abs/2403.12873)

Bulgu:
- 0-4 saat horizonlarda cloud motion, cloud growth/decay ve sky images NWP'den daha hızlı sinyal verebilir.
- Görüntüyü doğrudan kullanmak yerine cloud fraction/motion/variance gibi kompakt feature'lar da işe yarayabilir.

OpenEnergy senaryosu:

```text
solar_intraday_cloud_features =
  horizon:
    5min-4h
  sources:
    satellite cloud motion
    sky camera
    compact cloud features
  status:
    day-ahead MVP sonrası roadmap
```

Öncelik: P2 day-ahead için, P0 intraday ürün için.  
Risk: Veri entegrasyonu, latency, kamera kalitesi ve spatial alignment.

### S10 — Interpretable Probabilistic PV Modeli Operasyonel Güven İçin Faydalı

Kaynak: [NGBoost for interpretable probabilistic PV forecasting](https://arxiv.org/abs/2108.04058)

Bulgu:
- Probabilistic PV forecast ile SHAP/açıklanabilirlik birlikte kullanılabiliyor.
- Kullanıcı forecast hatasının cloud, spread, temperature veya regime kaynaklı olduğunu görmek ister.

OpenEnergy senaryosu:

```text
solar_explainability =
  per forecast:
    top feature drivers
    cloud/regime flag
    uncertainty reason
  per scenario:
    why this scenario wins
```

Öncelik: Orta-yüksek.  
Risk: İlk MVP'de raporlama ağırlaşabilir; önce scenario leaderboard kurulmalı.

## 4. Ortak En İyi Senaryo Aileleri

Aşağıdaki senaryo aileleri hem rüzgar hem solar için Strategy Search içinde explicit olarak denenmeli.

### Aile A — Physical Core + Tree Boosting

```text
features:
  physical core features
model:
  LightGBM / CatBoost / XGBoost
train:
  expanding_all veya rolling_365d
ensemble:
  none veya rank_weighted
```

Ne zaman iyi olabilir:
- Veri az/orta seviyedeyse.
- Santral davranışı stabilse.

### Aile B — NWP Ensemble/Spread + Quantile

```text
features:
  NWP mean/spread/min/max
  uncertainty features
model:
  QuantileGBM / LightGBM quantile / conformalized model
train:
  rolling_180d veya rolling_365d
ensemble:
  QRA / quantile calibration
```

Ne zaman iyi olabilir:
- Day-ahead ve 48h horizon.
- Belirsizlik bandı önemliyse.

### Aile C — Multi-Point Spatial

```text
features:
  nearest point
  average_3
  point spread
  spatial_stack
model:
  tree boosting veya stacking
train:
  rolling_180d
```

Ne zaman iyi olabilir:
- Büyük RES sahaları.
- Kaba NWP grid.
- Dağlık/karmaşık arazi.

### Aile D — Dynamic Bias Correction

```text
target:
  observed production veya forecast residual
features:
  recent errors
  NWP forecast
  temporal features
model:
  Ridge / LightGBM / CNN ileride
train:
  recent_weighted / rolling_90d / rolling_180d
```

Ne zaman iyi olabilir:
- NWP sistematik bias yapıyorsa.
- Santral davranışı drift ediyorsa.

### Aile E — Regime-Specific Strategy

```text
regime:
  solar: clear/cloudy/variable
  wind: low/medium/high wind, direction sector
per regime:
  ayrı feature set
  ayrı model
ensemble:
  regime gate
```

Ne zaman iyi olabilir:
- Solar için bulutluluk değişkense.
- Rüzgar için yön/wake etkisi güçlüyse.

### Aile F — Top-K Candidate Ensemble

```text
single candidates:
  farklı feature set + farklı model
top_k:
  validation'a göre seç
ensemble:
  simple_mean
  inverse_error_weighted
  optuna_weighted
  stacking_meta
```

Ne zaman iyi olabilir:
- Tek bir feature-model kombinasyonu tüm koşullarda kazanamıyorsa.
- Santral bazında farklı dinamikler varsa.

## 5. "En Çok Tutan Senaryolar" Hafızası

Kullanıcının istediği çıkarım için uygulama içinde bir **Scenario Memory** tutulmalı.

Amaç:

```text
Tüm experiment sonuçlarından:
  hangi senaryo aileleri,
  hangi feature grupları,
  hangi train policy'ler,
  hangi ensemble türleri
hangi santral tiplerinde daha çok kazanıyor?
```

### 5.1 Önerilen Tablolar

```sql
scenario_families (
  family_id,
  name,
  asset_kind,
  description,
  config_template_json
)

scenario_runs (
  scenario_run_id,
  experiment_id,
  plant_id,
  horizon_hours,
  family_id,
  strategy_id,
  feature_signature,
  model_signature,
  train_policy,
  point_policy,
  ensemble_policy,
  quantile_policy,
  nrmse,
  nmae,
  bias,
  pinball,
  coverage,
  skill_score,
  runtime_seconds,
  is_champion,
  created_at
)

scenario_leaderboard (
  asset_kind,
  horizon_hours,
  family_id,
  wins,
  trials,
  win_rate,
  median_skill_score,
  median_nrmse,
  median_pinball,
  last_updated_at
)

plant_strategy_profile (
  plant_id,
  horizon_hours,
  best_family_id,
  best_strategy_id,
  stable_winner_count,
  drift_detected,
  recommended_next_search_json,
  updated_at
)
```

### 5.2 Feature Signature

Feature setleri karşılaştırılabilir olmalı. Bunun için her candidate'a normalize edilmiş imza verilmeli:

```text
feature_signature =
  wind_power+air_density+direction+cyclical_time

model_signature =
  lightgbm:n_estimators_bucket=200-400:num_leaves_bucket=31-63

strategy_signature =
  wind_nwp_spread|rolling_180d|weighted_ensemble|conformal
```

### 5.3 Leaderboard Soruları

UI ve backend şu sorulara cevap vermeli:

- Bu plant için son 10 experiment'te hangi strategy ailesi kazandı?
- Rüzgar 24h horizon'da genel olarak en çok hangi feature grubu kazanıyor?
- Solar 48h horizon'da clear-sky index kullananlar mı kullanmayanlar mı daha iyi?
- Rolling 180d mi expanding_all mı daha çok champion oluyor?
- Ensemble tek modeli ne sıklıkla yeniyor?
- Multi-point policy gerçekten fark yaratıyor mu?
- p10/p90 calibration başarısız olan strategy'ler hangileri?

## 6. Strategy Search İçin Önerilen Presetler

### quick

```text
trial budget: düşük
feature: physical_core
model: LightGBM, Ridge, RandomForest
train: expanding_all
ensemble: none
```

### balanced

```text
feature: physical_core + uncertainty + temporal
model: LightGBM, XGBoost, CatBoost, RandomForest, QuantileGBM
train: expanding_all, rolling_180d
ensemble: simple_mean, inverse_error_weighted
```

### deep

```text
feature: tüm anlamlı gruplar
model: tüm model havuzu
train: rolling_90/180/365 + recent_weighted
ensemble: top-K weighted + stacking
quantile: conformal calibration
```

### research

```text
balanced/deep +
multi-point search
NWP source/spread search
AutoGluon/FLAML challenger
regime-specific strategy
```

## 7. Uygulama Önceliği

1. **Scenario Memory şemasını ekle**
   - En çok tutan senaryoları kaydetmek için temel şart.

2. **StrategyConfig'i trial kayıtlarına yaz**
   - Sonradan analiz edilemeyen experiment boşa gider.

3. **Train policy search**
   - Dynamic train etkisini hızlı görürüz.

4. **Top-K candidate ensemble**
   - Kullanıcının istediği feature grubu A + feature grubu B ensemble senaryosunun çekirdeği.

5. **NWP ensemble/spread features**
   - Literatürde en güçlü sinyal.

6. **Multi-point weather search**
   - Santral bazlı dinamikleri yakalama açısından kritik.

7. **Quantile calibration**
   - p10/p90 kalitesi için şart.

8. **AutoGluon/FLAML challenger**
   - AutoGluon Tabular: OpenEnergy feature matrix üstünde AutoML tabular ensemble.
   - AutoGluon TimeSeries/Foundation: Chronos-2, Chronos-Bolt, DirectTabular,
     RecursiveTabular gibi variant'lar target history + known covariates ile.
   - Hepsi Scenario Memory'de `engine`, `model_family`, `model_variant` ile ayrı izlenir.
   - Ana motorun altında yarışmacı olarak eklenmeli.

## 8. OpenEnergy İçin Net Sonuç

En iyi tahmin sonucunu bulmak için sistem şu şekilde davranmalı:

```text
1. Her plant+horizon için birçok candidate pipeline üret.
2. Candidate'ları leakage-safe validation ile değerlendir.
3. En iyi top-K candidate'ı sakla.
4. Top-K üstünden ensemble stratejileri dene.
5. Train policy'leri ayrı ayrı yarıştır.
6. Quantile calibration uygula.
7. Champion strategy'yi üretime al.
8. Her sonucu Scenario Memory'ye kaydet.
9. Zamanla hangi senaryoların hangi santral tiplerinde tuttuğunu öğren.
10. Yeni experiment başlarken geçmiş hafızadan iyi başlangıç öner.
```

Bu sayede OpenEnergy, yalnızca en iyi modeli değil, **santral bazında en çok tutan tahmin senaryosunu** öğrenen bir sistem olur.
