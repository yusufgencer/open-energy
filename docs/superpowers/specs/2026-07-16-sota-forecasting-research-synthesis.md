# OpenEnergy — Wind & Solar Forecasting SOTA & Best-Practice Synthesis

_Derleme: 2026-07-16. Kaynak: 6 paralel araştırma ajanı (fable) — rüzgar feature engineering,
rüzgar ensemble, solar feature engineering + tepe-yakalama, solar ensemble, horizon stratejisi,
NWP kaynakları. Ham raporlar scratchpad/research/ altında._

Bu doküman, mevcut mimari (tabular GBM + Optuna + DuckDB + Open-Meteo previous-runs, horizon
24/48, quantile p10/p50/p90) için literatür + yarışma (GEFCom2014, HEFTCom2024, Kaggle AMS,
EEM2020, M5-U) kanıtına dayalı en iyi uygulamaları özetler ve **çok-noktalı hava optimizasyonu
tasarımına** ve genel yol haritasına bağlar.

---

## 0. En güçlü, tekrar eden bulgular (tüm raporların kesişimi)

Altı bağımsız araştırmanın **yakınsadığı** çekirdek — en yüksek güven:

1. **Tabular GBM (LightGBM/XGBoost/CatBoost) + pinball loss quantile = day-ahead için SOTA.**
   Derin öğrenme tabular NWP feature'larda GBM'i geçmiyor. Mevcut mimarimiz doğru omurgada.
2. **En büyük tek kaldıraç model sınıfı değil, hava girdisi çeşitliliği**: çok-NWP-kaynağı ve
   çok-noktalı grid. Rüzgarda +6–17% MAE; solar için multi-source marjinal (~0.1–0.4%) ama
   multi-point yine değerli.
3. **Quantile hijyeni**: p10/p50/p90'ı her modelden ve her ensemble'dan sonra **sırala**
   (Chernozhukov rearrangement) — teorik olarak asla zarar vermez, bizde yok.
4. **Conformal kalibrasyon (CQR / adaptive)** ensemble p10/p90 üstüne, horizon başına, kayan
   pencerede → dağılımdan-bağımsız coverage garantisi. Bizde yok, ucuz kazanç.
5. **Per-quantile ensemble kombinasyonu (QRA / Vincentization)**: ağırlıklar quantile başına
   (p10 ağırlıkları ≠ p90 — asimetrik hatalar). Mevcut ensemble katmanımız p50 üstünden çalışıyor.
6. **Solar tepe-yakalama sorununun kök çözümü = clear-sky-index hedef dönüşümü** (kPV = P/P_cs
   tahmin et, geri çarp). Ağaçlar ekstrapole edemez; fizik zirveyi verir, GBM sadece [0,1] bulut
   zayıflatmasını öğrenir.
7. **"En iyi tek noktayı seç" naif sezgisi yanlış**: Andrade & Bessa 2017 (tam bizim stack)
   uzamsal **ortalama/PCA'nın tek noktayı geçtiğini** gösteriyor — spatial_mean/mean+spread
   single_best'ten üstün olmalı.
8. **Optuna bütçesini kısıtla**: sadece p50 modelini tune et, parametreleri tüm quantile'lara
   aktar (HEFTCom hilesi, ~3× tasarruf). Arama uzaylarını yarışma-doğrulanmış aralıklara sıkıştır.

---

## 1. Rüzgar — Feature Engineering

| Teknik | Ne / Neden | Bizde uygulanabilirlik |
|---|---|---|
| **Güç eğrisi dönüşümleri** ws100²/ws100³, clipped power-curve pc(v), cut-out flag | v³ tek en çok atıflı türetilmiş feature; NWP hatası ~3× büyür | Yeni "wind_physics" feature family — saf kolon dönüşümü |
| **ρ·v³ hava yoğunluğu düzeltmesi** (ρ = p/RT) | Çıktı ~lineer ρ ile; ~10% enerji farkı | pressure/temperature Open-Meteo'da var |
| **Çok-seviye rüzgar + shear** α = ln(ws120/ws10)/ln(120/10) | Stabilite proxy'si; güç eğrisini ~15% kaydırır | 10/80/100/120/180m çek |
| **NWP trajektori bağlamı** ws100 (ve pc) t±1,t±2,t±3 + rolling mean/std/min/max | GEFCom2014 kazananının temeli; sızıntısız (tüm trajektori issue-time'da mevcut) | En çok yarışma-doğrulanmış temporal teknik |
| **Yön kodlama** sin/cos(wd100), u/v bileşenleri, v·sin/v·cos etkileşimleri, 8–16 sektör | Ham derece 0/360 dikişinde ağaç bölmelerini kırar; yön tek başına +11% (kıyı/karmaşık) | CatBoost sektörü native işler |
| **Gust + pseudo-TI** gusts/ws, rolling std/mean, Δws ramp | Türbülans güç eğrisini düzler; kanıt karışık — doğrula | İkinci öncelik |

**En büyük sinyal**: dönüştürülmüş hub-yükseklik rüzgar hızı + zamansal komşuluğu ≈ %80.
Üretim lag'leri ≥24h'de neredeyse değersiz.

---

## 2. Rüzgar — Model & Ensemble

- **Algoritmalar**: XGB/LGBM/CatBoost ~eşit; LightGBM en hızlı. CatBoost ordered boosting
  <~100K satırda (per-plant 1–3 yıl saatlik = tam bu bölge) faydalı ama quantile desteği zayıf →
  p50/point için CatBoost, quantile başlıkları için LightGBM. Sığ ağaçlar hakim (depth 3–7,
  num_leaves 10–100, lr 0.01–0.1).
- **QRF (quantile regression forest)** EEM2020'yi kazandı — tüm quantile'lar inşaen çaprazlanmaz.
  RandomForest'ı QRF quantile çıkarımına yükselt.
- **Ensemble**: linear quantile kombinasyonu (Vincentization) tek modeli güvenilir geçer; farklı
  tümevarımsal önyargılı modelleri birleştir (Ridge/linear üyeyi solo CV zayıf olsa da tut).
  Caruana greedy selection = optuna_weights'e sağlam alternatif. Stacking meta: sadece OOF üstünde,
  non-negatif (NNLS), quantile-regression meta (kare-hata Ridge değil).
- **Curtailment hijyeni**: kısıtlı saatleri eğitimden çıkar (0.5 m/s rüzgar-bin 5. persentil altı;
  ~3–6% veri). Yoksa güç eğrisi aşağı sapar.
- **Hibrit**: saf ML saf fiziği kesin geçer; fiziği FEATURE olarak kullan (pc, ρ·v³), baseline değil.
  Kısa geçmişli santraller için residual-vs-powercurve hedef.

---

## 3. Solar — Feature Engineering + ★ Tepe-Yakalama

**Irradiance**: fiziksel sürücü GHI değil POA/GTI. Open-Meteo `global_tilted_irradiance` (tilt/azimuth
ile), DNI, DHI, diffuse fraction (DHI/GHI = bulut rejimi feature'ı) çek.

**Solar geometri** (pvlib, deterministik, her horizon'da sızıntısız): cos_zenith (clear-sky GHI'ye ≈
orantılı, tek en güçlü), elevation, sin/cos azimuth, air mass, extraterrestrial, day length,
hours_since_sunrise/to_sunset. Ham hour/doy cyclical'ı geçer.

**★ TEPE-YAKALAMA SORUNU** — GBM'ler öğle zirvesini sistematik olarak düşük tahmin eder:
- **Nedenler**: (1) ağaçlar ekstrapole edemez (yaprak = ortalama, clear+near-clear karışımı clear-sky
  altına ortalanır); (2) simetrik L2/L1 + çarpık koşullu dağılım → koşullu ortalama mode'un altında;
  (3) NWP smoothing yüksek irradiance'ta düşük yanlı; (4) zirve saatleri az sayıda örnek.
- **Sıralı çözümler**:
  1. **kPV hedef dönüşümü (EN YÜKSEK ETKİ)**: `y' = P/P_cs` üstünde eğit, `ŷ'×P_cs` çıkar. Fizik
     zirve zarfını verir, GBM sadece [0,1] bulut zayıflatmasını öğrenir — ekstrapolasyon yok.
     Quantile'lar doğal: kPV quantile × P_cs → açık günde dar, değişken öğlede geniş bant.
  2. **Model-chain feature'ı** P_physical (GTI→temp→PVWatts) feature olarak ve/veya residual hedef.
     5.2–10.4% MAE düşüşü (Mayer 2022, ≥1yıl veri gerekir).
  3. **Gündüz-maskeleme** (zenith>85–89° at) + gece hard sıfır + P_cs-orantılı sample weight.
  4. **Asimetrik loss** (yalnız #1–3 artık bias bırakırsa; calibration'ı bozabilir).
  5–8. Quantile-shifted point, regime modelleri, capacity/inverter clipping cap, uzamsal grid std.
- **Clear-sky modeli**: Ineichen-Perez yeterli — daha süslü (REST2) forecast avantajı YOK (Yang).
- **Sıcaklık derating**: Faiman cell temp (POA+T2m+wind), γ≈−0.3…−0.45%/°C. Snow depth ekle.

**Not**: kPV/POA yolu **pvlib** bağımlılığı getirir (şu an pyproject'te yok).

---

## 4. Solar — Model & Ensemble

- Clear-sky-index alanında tahmin ensemble ağırlıklarını da stabilize eder (ham-güç ağırlıkları
  öğle hatalarınca domine).
- **Multi-NWP sister model + per-quantile stacking** = yarışma-kazanan. HEFTCom2024 kazananı:
  NWP kaynağı başına CatBoost + meta. **Ama solar için multi-source kazancı ~0.1–0.4% (marjinal)** —
  bütçeyi clear-sky-index + bulut feature'larına harca.
- **CatBoost MultiQuantile** (tek model, 9 quantile birlikte — çaprazlanmayı da atlar) model tipi ekle.
- **QRA (quantile regression averaging)** = standart combiner; stacking_meta_ridge'i per-quantile
  QRA'ya yükselt.
- **Forecast combination puzzle**: öğrenilen ağırlıklar OOS'ta genelde eşit ağırlığı kaybeder →
  `mean`'i champion baseline tut, learned weights'i ≥3 fold'da geçmesini şart koş, simplex + uniform'a
  shrink.
- **Seed-bagging** (3–5 seed ortalaması) = ucuz alfa.

---

## 5. Horizon Stratejisi

- **24/48h = NWP-driven ML rejimi** (üretim lag'i ~sıfır sinyal). Mevcut yaklaşım doğru.
- **Per-horizon (mevcut)** SOTA-tutarlı ama N× maliyet + cross-horizon tutarsızlık. **Gelecek evrim:
  tek pooled GBM + `lead_time_hours` feature + per-lead-bucket quantile kalibrasyon** — 3-saatlik
  yoğun grid'e model patlaması olmadan ölçeklenir, quantile bandı kısa lead'de otomatik daralır.
- **Dinamik 3-saatlik rejim**: reforecast tetikleyici = NWP run gelişi (saat değil). (init_time,
  lead_time) provenance sakla. Latest run'ı önceki ile blend et (~0.7/0.3) → jumpiness azalt.
- **"24 katı" kısıtını kır**: Open-Meteo **Single Runs API** (`run=`) keyfi lead'i sızıntısız verir
  (~€99/mo). Ara çözüm: previous_day3–7 ile 72–168h horizon'ları BUGÜN ekle (sıfır altyapı), her
  yeni horizon'u climatology-üstü-skill kapısına bağla.
- **Skill/baseline**: horizon başına skill raporla — rüzgar: persistence+climatology konveks;
  solar: **smart persistence** (clear-sky index persist) + climatology. Düz persistence solar için
  hatalı baseline.
- **Retraining/drift**: aylık base retrain + **günlük hafif LASSO polinom quantile post-processor**
  (HEFTCom deseni — capacity drift/bias'ı ucuza emer). Drift-tetikli acil retrain. Per-horizon
  champion-challenger + Diebold-Mariano anlamlılık.

---

## 6. NWP Kaynakları (Open-Meteo)

- **Varsayılanlar** — rüzgar: **ECMWF IFS (9km) + ICON-EU**; solar: **IFS varsayılan, ICON-EU
  challenger, GFS üçüncü çeşitlilik**. ICON-EU (6.5km, 3-saatlik) Türkiye'yi kapsayan tek sub-10km
  model. AROME/HARMONIE/UKV Türkiye'yi KAPSAMAZ.
- **Multi-source blend**: rüzgar için KESİN yap (paralel per-source feature, source-subset Optuna
  categorical); solar için deprioritize.
- **Çok-noktalı grid** (Andrade & Bessa 2017 = tam bizim stack): tek noktaya karşı MAE −16% solar /
  −13% rüzgar; iyi temporal model üstüne +4–6%; kısa geçmişte ~6%. **Uzamsal ortalama/PCA tek noktayı
  geçer.** Spatial std = belirsizlik rejim flag'i.
  - **Grid: NWP native çözünürlüğüne göre aralık** (oversample etme — Open-Meteo interpole eder).
    ICON-EU ~0.0625°(7km), IFS ~0.1°, GFS 0.25°. **5×5 grid native aralıkta (~30–50km kutu)** =
    25 nokta faydanın çoğunu yakalar. Rüzgar için hakim upwind yöne uzat.
- **Open-Meteo**: previous_dayN = N×24h önce tahmin (day1=24h, day2=48h, sızıntısız). **Arşiv tabanı
  Ocak 2024 (~2.5yıl)** — CV'yi buna göre planla. best_match/seamless'tan kaçın (hangi model
  ürettiğini gizler).
- **MOS gereksiz**: GBM NWP→güç eğitilince her kaynağın koşullu bias'ını örtük öğrenir. Sadece
  aynı lead-time'da eğit + periyodik retrain + rolling recent-error feature.

---

## 7. Çok-Noktalı Hava Optimizasyonu Tasarımına Doğrudan Etkiler

Bu araştırma, tasarlanan `point_policy` arama eksenini **yeniden şekillendiriyor**:

1. **point_policy set'ini uzamsal-agregasyona kaydır**: `single_best` muhtemelen düşük performans
   (literatür net) — yine de baseline/negatif kontrol olarak tut. `spatial_mean` ve `mean+spread`
   birincil; **`pca_k` (grid×değişken üstünde PCA) ekle** — rüzgar için en iyi uzamsal yöntem.
2. **Grid üretimi native çözünürlüğe kilitlensin**: kaynağa göre aralık (ICON-EU 0.0625°, IFS 0.1°).
   Varsayılan **5×5**; oversample redundant.
3. **spatial_std'i quantile bandı feature'ı yap** — hem rüzgar hem solar için belirsizlik rejim
   sinyali (p10/p90 genişliği).
4. **`source_policy`'yi paralel eksen olarak ekle** (ecmwf | icon_eu | gfs | ecmwf+icon_eu): rüzgarda
   güçlü, solarda deprioritize. scenario_runs zaten `point_policy` kolonu taşıyor; `source_policy`
   için de yer aç.
5. **NWP trajektori (t±1,t±2,t±3) + rolling** feature'ları grid-agregasyonuyla birlikte gelsin —
   tek başına en değerli feature sınıfı.
6. **scenario memory leaderboard**: "hangi nokta/kombinasyon en iyi" sorusu artık
   {point_policy × source_policy × grid_radius} üstünden yanıtlanır — Optuna categorical'ları olarak,
   25 noktayı tek tek aramak yerine (gürültülü).

Bu, kullanıcının orijinal isteğine ("lat/long'a göre birkaç noktadan hava verisi alıp optimizasyon")
literatür-doğrulanmış bir şekil verir: **nokta seçimi değil, uzamsal füzyon** ana kazançtır.

---

## 8. Genel Yol Haritası Fırsatları (çok-noktanın ötesinde, bu araştırmadan çıkan)

Öncelik sırasına yakın (her biri ayrı dilim olabilir):

1. **Quantile sıralama** (her model + ensemble sonrası) — trivial, asla zarar vermez. [tüm raporlar]
2. **Çok-noktalı + çok-kaynaklı hava** (mevcut tasarım) — en büyük doğrulanmış kazanç.
3. **Solar kPV hedef dönüşümü + gündüz maskeleme + peak metrikleri** — tepe-yakalamayı kökten çözer.
4. **Conformal kalibrasyon (CQR/ACI)** ensemble p10/p90 üstüne, horizon başına.
5. **Per-quantile ensemble (QRA/Vincentization)** — ensemble katmanını quantile başına çalıştır.
6. **Solar geometri + irradiance feature aileleri** (pvlib) + rüzgar fizik/trajektori aileleri.
7. **Skill-score değerlendirme** (doğru baseline'lara karşı) + peak/coverage/width metrikleri, gündüz.
8. **Günlük LASSO post-processor + drift-tetikli retrain** (retraining/champion.py'yi genişletir).
9. **Pooled + lead_time modeli + Single Runs API** → dinamik 3-saatlik rejim, day 3–7 horizon.
10. **CatBoost MultiQuantile + QRF + seed-bagging** model havuzu zenginleştirmeleri.

---

## Ana Kaynaklar

- **Andrade & Bessa 2017** — Improving Renewable Energy Forecasting with a Grid of NWP (IEEE TSE) — çok-nokta referansı
- **HEFTCom2024** — arXiv:2505.10367 (GEB) + arXiv:2507.01579 (overview) — sister-NWP stacking, per-quantile LightGBM
- **Landry et al. 2016** — GEFCom2014 wind kazananı (quantile GBM)
- **Bruninx et al. 2026** — arXiv:2602.13010 — tree-based + weather ensembles, CQR, curtailment
- **Mayer 2022** — RSER — PV hibrit fiziksel+ML (−5.2/−10.4% MAE)
- **Yang** — clear-sky model/index seçimi; smart persistence baseline
- **Seamless probabilistic wind** — arXiv:2502.11960 — pooled + lead-time, 6–162h
- **EEM2020** — QRF kazananı; Browell quantile combination
- **Kaggle AMS 2013–14** — GBRT + GEFS grid features
- Open-Meteo: [Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) · [Single Runs](https://openmeteo.substack.com/p/single-runs-api) · [Ensemble](https://open-meteo.com/en/docs/ensemble-api)
