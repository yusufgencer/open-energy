import openenergy.integrations.epias_store as store


class _FakeSettings:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.epias_username = None
        self.epias_password = None

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


def test_set_status_persist_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "get_settings", lambda: _FakeSettings(tmp_path))
    store.clear_credentials()
    assert store.status() == {"connected": False, "username": None}

    store.set_credentials("yusuf@example.com", "secret")
    st = store.status()
    assert st == {"connected": True, "username": "yusuf@example.com"}
    # status şifreyi sızdırmamalı
    assert "password" not in st
    # dosyaya yazıldı mı
    assert (tmp_path / "epias_credentials.json").exists()

    # belleği sıfırla → dosyadan geri yüklensin
    store._memory = None
    assert store.status()["connected"] is True

    store.clear_credentials()
    assert store.status()["connected"] is False
    assert not (tmp_path / "epias_credentials.json").exists()


def test_falls_back_to_settings_env(tmp_path, monkeypatch):
    s = _FakeSettings(tmp_path)
    s.epias_username = "envuser@example.com"
    s.epias_password = "envpass"
    monkeypatch.setattr(store, "get_settings", lambda: s)
    store._memory = None
    assert store.status() == {"connected": True, "username": "envuser@example.com"}
    store._memory = None
