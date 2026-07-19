# OpenEnergy — Strategy Search Engine Spec

**Tarih:** 2026-06-30
**Durum:** V1 implementation target

## Amaç

Mevcut tekil pipeline AutoML akışını, horizon başına açıkça kayıtlı ve leakage-safe
bir tahmin stratejisi aramasına dönüştürmek. Strategy search; model ailesi,
feature blokları, train policy ve top-K ensemble metodlarını birlikte dener.

## Kavramlar

- **CandidateConfig:** Tek feature-model-train policy kombinasyonu.
- **StrategyConfig:** Bir veya daha fazla candidate ile bir ensemble metodunun birleşimi.
- **TrainPolicy:** Eğitim satırlarının nasıl seçileceği/ağırlıklandırılacağı.
- **CandidateResult:** Candidate'ın CV ve outer-test skorları.
- **StrategyResult:** Ensemble strategy'nin CV proxy ve outer-test skorları.

## V1 Kapsamı

- Tek plant, tek point ve tek NWP source üzerinde çalışır.
- Feature block search mevcut leakage-safe raw row universe üzerinde yapılır.
- Train policy arama eksenleri: `expanding_all`, `rolling_90d`, `rolling_180d`,
  `rolling_365d`, `recent_weighted`.
- Ensemble metodları: `none`, `mean`, `weighted_by_cv`, `optuna_weights`,
  `stacking_meta_ridge`.
- Champion, strategy artifact olarak kaydedilir; eski single-model artifact yolu
  fallback olarak korunur.

## V2 Dışında Bırakılanlar

- Multi-point weather search.
- Multi-NWP gerçek kaynak seçimi ve NWP spread feature'ları.
- Quantile calibration / conformal post-processing.
- AutoGluon/FLAML gibi harici AutoML motorları.
