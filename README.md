# Open Energy

Open source enerji projesi — **FastAPI** (Python) backend, **React** frontend.

## Branch yapısı

| Branch | Açıklama |
|--------|----------|
| **main** | Production: sadece release edilmiş, stabil kod. Doğrudan commit atılmaz. |
| **develop** | Geliştirme: tüm özellikler burada birleşir. Günlük geliştirme bu dalda yapılır. |

**Akış:**
- Yeni özellikler: `develop`'dan `feature/özellik-adı` aç → bitince `develop`'a merge.
- Bugfix: `develop`'dan `fix/açıklama` aç → bitince `develop`'a merge.
- Release: `develop` hazır olunca `main`'e merge (ve gerekirse tag ile sürüm işaretle).

## Teknoloji

- **Backend:** Python, FastAPI
- **Frontend:** React
- **Veritabanı:** (henüz karar verilmedi)
