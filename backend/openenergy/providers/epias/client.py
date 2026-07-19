"""EPİAŞ Şeffaflık Platformu (EXIST Transparency v2) istemcisi.

Auth akışı (2025-12 sonrası, body'de credential):
  POST https://giris.epias.com.tr/cas/v1/tickets
       Content-Type: application/x-www-form-urlencoded
       body: username=...&password=...
  -> düz metin "TGT-...." (≈2 saat geçerli)
Veri istekleri: header `TGT: TGT-....` ile
  https://seffaflik.epias.com.tr/electricity-service/v1/...

Credential'lar EPIAS_USERNAME / EPIAS_PASSWORD ortam değişkenlerinden okunur
(ya da kurucuya verilir). Şifre asla loglanmaz.
"""
from __future__ import annotations

import os
import time

import httpx

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
BASE_URL = "https://seffaflik.epias.com.tr/electricity-service/v1"
# TGT 2 saat geçerli; limitin altında, güvenli bir yenileme penceresi.
_TGT_TTL_SECONDS = 105 * 60
_RETRY_STATUS = {429, 500, 502, 503, 504}


class EpiasAuthError(RuntimeError):
    """TGT alınamadı / credential eksik."""


class EpiasClient:
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        base_url: str = BASE_URL,
        cas_url: str = CAS_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        clock=time.monotonic,
    ) -> None:
        self.username = username or os.environ.get("EPIAS_USERNAME")
        self.password = password or os.environ.get("EPIAS_PASSWORD")
        self.base_url = base_url.rstrip("/")
        self.cas_url = cas_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._clock = clock
        self._tgt: str | None = None
        self._tgt_ts: float = 0.0

    # ── Auth ────────────────────────────────────────────────────────────────
    def _ensure_tgt(self) -> str:
        if self._tgt and (self._clock() - self._tgt_ts) < _TGT_TTL_SECONDS:
            return self._tgt
        if not self.username or not self.password:
            raise EpiasAuthError(
                "EPİAŞ credential yok: EPIAS_USERNAME ve EPIAS_PASSWORD ayarla "
                "(kayit.epias.com.tr ücretsiz hesabı)."
            )
        with httpx.Client(timeout=self.timeout) as c:
            resp = c.post(
                self.cas_url,
                data={"username": self.username, "password": self.password},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/plain",
                },
            )
        if resp.status_code not in (200, 201):
            raise EpiasAuthError(f"TGT alınamadı: HTTP {resp.status_code}")
        tgt = (resp.text or "").strip()
        if not tgt.startswith("TGT"):
            # Bazı CAS sürümleri TGT'yi Location header'ında döndürür.
            loc = resp.headers.get("Location", "")
            tgt = loc.rsplit("/", 1)[-1] if loc else tgt
        if not tgt.startswith("TGT"):
            raise EpiasAuthError("TGT yanıtı beklenen formatta değil")
        self._tgt = tgt
        self._tgt_ts = self._clock()
        return tgt

    def _headers(self) -> dict:
        return {
            "TGT": self._ensure_tgt(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ── Veri istekleri ──────────────────────────────────────────────────────
    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        with httpx.Client(timeout=self.timeout) as c:
            for attempt in range(self.max_retries + 1):
                resp = c.request(method, url, headers=self._headers(), json=json)
                if resp.status_code in _RETRY_STATUS and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
        raise AssertionError("unreachable: retry loop always returns or raises")

    def get(self, path: str) -> dict:
        return self._request("GET", path)

    def post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, json=body)
