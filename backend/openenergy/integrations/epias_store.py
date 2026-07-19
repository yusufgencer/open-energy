"""EPİAŞ credential store — UI'den girilen kullanıcı adı/şifreyi tutar.

Lokal/tek kullanıcı senaryosu: credential'lar process belleğinde tutulur ve
opsiyonel olarak `data/epias_credentials.json` dosyasına (0600, gitignore'lu)
yazılır → restart'ta korunur. Şifre asla loglanmaz; status'ta gösterilmez.

Öncelik sırası: bellek > kayıtlı dosya > Settings(.env). Böylece .env hâlâ çalışır.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openenergy.config import get_settings
from openenergy.providers.epias.client import EpiasAuthError, EpiasClient


@dataclass
class _Creds:
    username: str
    password: str


_memory: _Creds | None = None


def _store_path():
    return get_settings().data_dir / "epias_credentials.json"


def _load_file() -> _Creds | None:
    path = _store_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return _Creds(username=data["username"], password=data["password"])
    except (ValueError, KeyError, OSError):
        return None


def _current() -> _Creds | None:
    global _memory
    if _memory is not None:
        return _memory
    f = _load_file()
    if f is not None:
        _memory = f
        return f
    s = get_settings()
    if s.epias_username and s.epias_password:
        return _Creds(username=s.epias_username, password=s.epias_password)
    return None


def set_credentials(username: str, password: str, *, persist: bool = True) -> None:
    global _memory
    _memory = _Creds(username=username, password=password)
    if persist:
        settings = get_settings()
        settings.ensure_dirs()
        path = _store_path()
        path.write_text(json.dumps({"username": username, "password": password}))
        try:
            os.chmod(path, 0o600)  # yalnızca sahibi okuyabilsin
        except OSError:
            pass


def clear_credentials() -> None:
    global _memory
    _memory = None
    path = _store_path()
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def status() -> dict:
    creds = _current()
    if creds is None:
        return {"connected": False, "username": None}
    return {"connected": True, "username": creds.username}


def get_client() -> EpiasClient:
    creds = _current()
    if creds is None:
        raise EpiasAuthError("EPİAŞ bağlı değil — önce kullanıcı adı/şifre ile bağlan.")
    return EpiasClient(username=creds.username, password=creds.password)


def test_connection(username: str, password: str) -> None:
    """TGT almayı dener; başarısızsa EpiasAuthError fırlatır."""
    client = EpiasClient(username=username, password=password)
    client._ensure_tgt()  # noqa: SLF001 — kasıtlı: bağlantı testi
