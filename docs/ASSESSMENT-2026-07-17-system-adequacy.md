# OpenEnergy — Sistem Yeterlilik Değerlendirmesi

_2026-07-17. 5 bağımsız denetim subagent'ının (metodoloji, kod/test, gerçek-veri/pipeline,
mimari/ops, ürün/UX) eleştirel, kanıt-temelli bulgularının sentezi._

## Genel verdict

**Araştırma/backtest prototipi olarak: sağlam temel (B-). Üretim tahmin servisi olarak: yetersiz (D+).**

Sistem, sofistike ve iyi test edilmiş düşük-seviye bileşenlerle (conformal, splits, metrics,
per-quantile LP, quantile monotonluk) ve **gerçekten leakage-safe bir eğitim pipeline'ı** ile
güçlü bir araştırma motoru. Ancak: (1) **öne çıkan birkaç özellik gerçek/üretim kod yolunda
bozuk** ve testleri sahte, (2) **ileriye-dönük servis yolu kopuk**, (3) **sonuçlar iyimser ve
istatistiksel olarak güçsüz**, (4) **operasyonel katman ve ölçek hazır değil**.

| Boyut | Not | Tek cümle |
|---|---|---|
| Metodoloji & istatistik | **C+** | Eğitim leak-safe; ama champion + kalibrasyon **test setinde seçiliyor**; olasılıksal katman yapısal olarak eksik-kapsamlı. |
| Kod doğruluğu & test | **C−** | 3 flagship özellik gerçek yolda bozuk + testleri sahte; 2 arama ekseni hiç bağlı değil. |
| Gerçek-veri & sonuçlar | **C+** | Eğitim pipeline B+; ama sonuçlar ~1000 satır / tek 9-günlük blok / peek edilmiş → güvenilir değil. |
| Mimari & ops | **D** | Offline motor güçlü; üretim servis yolu bağlı değil; scheduler yok; jobs/concurrency ölçeklenmez. |
| Ürün & UX | **C−** | A-sınıfı onboarding, F-sınıfı operasyonel katman. Model kanıtlanabiliyor ama çalıştırılamıyor/izlenemiyor. |

## Yakınsayan kritik bulgular (birden çok denetimin bağımsız bulduğu = yüksek güven)

### 1. Selection-on-test → skill'ler iyimser üst-sınır (3 denetim)
`runner.py:479-490` ensemble metodunu **`y_test` üstünde min nRMSE** ile seçiyor; `quantile_policy`
de test coverage'ıyla seçiliyor (`strategy.py:503-539`). Aday hiperparametreleri iç-CV ile dürüst
seçiliyor **ama** ensemble/kalibrasyon katmanı test penceresinde seçiliyor → raporlanan skill
(0.14–0.47) ve coverage yukarı yanlı. Kanıt: kazanan ensemble santraller arası **rastgele**
(none/mean/optuna/vincentization/stacking) = 210 noktada gürültüye-fit imzası. **Düzeltme:
metod + policy'yi validation'da seç, test'i yalnız raporlama için ayır.**

### 2. İleriye-dönük servis yolu KOPUK (2 denetim)
Hiçbir ingest FORECAST-role hava yazmıyor (`fetch_forecast` çağrılmıyor) → `forecasts` tablosu
boş → `generate_forecast` 0 döner → `reforecast rows=0` → drift sonsuza dek `insufficient_data`
→ günlük LASSO post-processor'ın (forecast,actual) çifti yok. **Sadece backtest çalışıyor.** UI'da
"İleriye Dönük" sekmesi gerçek santrallerde kalıcı boş. **Bugün ürün: "model geçmişte nasıl yapardı",
"yarının üretimi" değil.**

### 3. Olasılıksal katman güvenilmez (2 denetim)
~66% coverage yapısal: dar native quantile + quantile-ortalama daraltması + val→test conformal
exchangeability kırık + **regime-binned CQR champion yolunda ölü kod** (`strategy.py:491-499` bin
yok; sadece pooled kullanıyor). Policy test coverage'ıyla seçilse bile 66%. Trading/dispatch için
tehlikeli (ramp kaçırma: yuntdag 07-13T18 actual 17.5 ama p10 36.6). **80% hedefi hiçbir yerde
assert/gate edilmiyor.**

### 4. Ölçek hazır değil (2 denetim)
DuckDB tek-yazar/in-process; 12 plant "gerçek zaman aldı", 303 = ~25× → API'yi bloklar. Open-Meteo
303×9=2727 çağrı/refresh, **backoff yok** → 429. Scheduler yok — "aylık/drift-tetikli" = insan curl'ü
plant×horizon başına. Jobs efemer (FastAPI BackgroundTasks, retry/crash-recovery/queue yok).

## Bozuk gerçek-yol özellikleri (kod denetimi)

- **[KRİTİK] kPV serving yanlış uzayda**: `predict_serving` adayları per-candidate kPV inverse
  OLMADAN birleştiriyor → karışık kpv+capacity_norm ensemble solar'da sabah/akşam ramp'te ~10× aşırı
  tahmin. (Rüzgar RES'ler etkilenmez — latent.) Testi sahte (capacity_norm modelde target_policy
  monkey-flip).
- **[YÜKSEK] champion defend servis yolunda bypass**: retrain `strategy_trials.is_champion`'a
  dokunmuyor; serving strategy champion'ı tercih ediyor → **reddedilen challenger yine servis
  ediliyor** → DM kapısı forecast yolu için no-op.
- **[ORTA-YÜKSEK] reforecast blend sessiz no-op**: aynı horizon'da new/old eşleştiriyor → valid_time
  kümeleri ayrık → kesişim boş → `updated=0` ama `blended=True`. Testi fiziksel-imkansız satırla
  geçiyor. **Düzeltme: valid_time'a göre horizon'lar-arası eşleştir.**
- **[ORTA] 2 ölü eksen**: `source_policy` `suggest`'e hiç geçirilmiyor → hep `nwp_sources[0]`;
  kaydedilen `source_policy` kozmetik, gerçek `source=` combo'yla **çelişebilir** → leaderboard yanlış
  etiketler. Pooled competitor **hiç çağrılmıyor**. İkisi de üretimde ölü. **(NOT: RES operasyonunda
  raporlanan `source_policy=icon_eu` gerçek bir arama değildi — düzeltme.)**

## Gerçek güçlü yanlar (adil olmak için)

- **Leakage-safe eğitim pipeline'ı** — tüm denetimler hemfikir; en güçlü kısım (role=previous_runs
  AND lead_hours=horizon; trajectory forecast serisinde, AR target leak yok; group_time_split; inner
  embargo). Gerçek market gate'ine karşı **konservatif** (fazladan skill kullanmıyor).
- **point_policy araması gerçek ve etkili** — uzamsal füzyon tek noktayı gerçek veride geçti
  (`all_points+spatial_mean+single_best` her santralde kazandı).
- İyi test edilmiş düşük-seviye bileşenler (conformal, splits, metrics, per-quantile LP, DM testi).
- DM-kapılı champion-challenger mantığı (offline), idempotent migration, ücretli API off-by-default.
- **A-sınıfı onboarding UI** (create→ingest→experiment→backtest, tam progress/empty/error).

## Öncelikli düzeltme yol haritası (en yüksek kaldıraç önce)

1. **Selection-on-test'i öldür** — ensemble metod + quantile policy validation'da seçilsin; skill
   gerçekten held-out pencerede raporlansın. (En büyük skill şişmesini kaldırır, coverage'ı
   dürüstleştirir.)
2. **Servis yolunu bağla** — FORECAST-role ingest job'u (`fetch_forecast`, NWP-run cadence, per-issue
   uniqueness) → forward forecast + drift + post-processor **gerçekten çalışır**.
3. **Bozuk özellikleri düzelt** — reforecast blend (valid_time/horizon-arası), champion defend
   (strategy_trials), kPV serving inverse, source_policy + pooled'ı bağla ya da sil.
4. **Kalibrasyonu 80%'e getir** — champion yolunda regime-binned CQR aç; adanmış geç conformal blok;
   quantile-ortalama yerine olasılık-ortalama.
5. **Belirsizliği raporla** — rolling-origin (≥3 fold) + DM anlamlılık + bootstrap CI; çok-mevsim veri
   (≥6-12 ay); curtailment/outage saatlerini temizle (EPİAŞ `total` ham sayaç).
6. **Ölçek için yeniden mimari** — out-of-process worker + scheduler + Open-Meteo batching/backoff;
   yazma yükünü API'den ayır; model registry + GC + versiyon damgası.
7. **Operasyonel UI yüzeyleri** — fleet dashboard, drift/monitoring, forward forecast tetikleme,
   ölü api.ts fonksiyonlarını (scenario/strategy leaderboard) bağla.

## Sonuç

Motor **metodolojik temeli ve mühendisliği sağlam bir araştırma prototipi** — ama şu an bir üretim
tahmin servisi DEĞİL. 0.14–0.47 skill'ler **umut verici bir smoke test**, doğrulanmış bir yetenek
değil (tek 9-günlük blok, test-seçili konfig, eksik-kapsamlı bantlar). En yüksek kaldıraçlı adım
#1 (selection-on-test) ve #2 (servis yolu): ilki sonuçları dürüstleştirir, ikincisi ürünü gerçekten
tahmin üreten hale getirir.
