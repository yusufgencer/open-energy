import datetime as dt

from openenergy.assets.repository import Plant, create_plant
from openenergy.ingestion.epias import ingest_epias_aic, ingest_epias_dpp


class SignalClient:
    def __init__(self):
        self.aic = 8.0
        self.dpp = 6.0
        self.get_calls = 0

    def get(self, path):
        self.get_calls += 1
        assert path == "/generation/data/powerplant-list"
        return {"items": [{"id": 42, "name": "Rüzgar Bir RES"}]}

    def post(self, path, body):
        assert body["powerPlantId"] == 42
        if path == "/generation/data/aic":
            return {"items": [{
                "date": "2026-06-01T03:00:00+03:00",
                "aic": self.aic,
            }]}
        if path == "/generation/data/dpp":
            return {"items": [{
                "date": "2026-06-01T03:00:00+03:00",
                "dpp": self.dpp,
            }]}
        raise AssertionError(f"unexpected path: {path}")


def test_store_and_reingest_aic_is_idempotent(db):
    create_plant(
        db, Plant(plant_id="wf1", name="Rüzgar Bir", kind="wind", capacity_mw=10)
    )
    client = SignalClient()
    kwargs = {
        "start": dt.date(2026, 6, 1),
        "end": dt.date(2026, 6, 1),
        "client": client,
    }

    first = ingest_epias_aic(db, "wf1", **kwargs)
    client.aic = 7.5
    second = ingest_epias_aic(db, "wf1", **kwargs)

    assert first["power_plant_id"] == second["power_plant_id"] == 42
    assert client.get_calls == 1  # second ingest reuses the persisted mapping
    assert db.execute(
        "SELECT ts, aic_mw FROM epias_aic WHERE plant_id='wf1'"
    ).fetchall() == [(dt.datetime(2026, 6, 1, 0), 7.5)]
    assert db.execute(
        "SELECT power_plant_id, power_plant_name FROM epias_plant_mapping "
        "WHERE plant_id='wf1'"
    ).fetchone() == (42, "Rüzgar Bir RES")
    assert db.execute("SELECT count(*) FROM production").fetchone()[0] == 0


def test_store_and_reingest_dpp_is_idempotent(db):
    create_plant(
        db, Plant(plant_id="wf1", name="Rüzgar Bir", kind="wind", capacity_mw=10)
    )
    client = SignalClient()
    kwargs = {
        "start": dt.date(2026, 6, 1),
        "end": dt.date(2026, 6, 1),
        "power_plant_id": 42,
        "plant_name": "Rüzgar Bir RES",
        "client": client,
    }

    ingest_epias_dpp(db, "wf1", **kwargs)
    client.dpp = 5.5
    result = ingest_epias_dpp(db, "wf1", **kwargs)

    assert result["rows"] == 1
    assert db.execute(
        "SELECT ts, dpp_mw FROM epias_dpp WHERE plant_id='wf1'"
    ).fetchall() == [(dt.datetime(2026, 6, 1, 0), 5.5)]
    assert db.execute("SELECT count(*) FROM production").fetchone()[0] == 0
