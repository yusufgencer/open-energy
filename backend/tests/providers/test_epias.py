import datetime as dt
import json

import httpx
import polars as pl
import pytest
import respx

from openenergy.providers.epias.client import BASE_URL, CAS_URL, EpiasClient
from openenergy.providers.epias.generation import (
    fetch_aic,
    fetch_dpp,
    fetch_realtime_generation,
    find_powerplant,
    list_powerplants,
)


class FakeClient:
    def __init__(self, powerplants=None, generation=None, responses=None):
        self._powerplants = powerplants or {}
        self._generation = generation or {}
        self._responses = responses or {}
        self.last_path = None
        self.last_body = None

    def get(self, path):
        return self._powerplants

    def post(self, path, body):
        self.last_path = path
        self.last_body = body
        if path in self._responses:
            return self._responses[path]
        return self._generation


def test_find_powerplant_case_insensitive_substring():
    client = FakeClient(powerplants={"items": [
        {"id": 1216, "name": "ÇATALCA RES", "eic": "X"},
        {"id": 2050, "name": "BARES BALIKESIR RES", "eic": "Y"},
    ]})
    assert list_powerplants(client)[0]["id"] == 1216
    m = find_powerplant(client, "bares")
    assert m is not None and m["id"] == 2050


def test_fetch_realtime_generation_parses_and_converts_tz():
    # EPİAŞ +03:00 saatleri → UTC-naive; total alanı power_mw olur
    client = FakeClient(generation={"items": [
        {"date": "2026-06-01T00:00:00+03:00", "total": 30.0, "wind": 30.0},
        {"date": "2026-06-01T01:00:00+03:00", "total": 42.5, "wind": 42.5},
        {"date": "2026-06-01T02:00:00+03:00", "total": None},  # atlanır
    ]})
    df = fetch_realtime_generation(client, 2050, dt.date(2026, 6, 1), dt.date(2026, 6, 2))
    assert df.height == 2
    # 00:00 +03:00 == 21:00 UTC önceki gün
    assert df["ts"][0] == dt.datetime(2026, 5, 31, 21, 0)
    assert df["power_mw"][0] == 30.0
    # istek gövdesi doğru powerPlantId ve +03:00 tarih içeriyor mu
    assert client.last_body["powerPlantId"] == 2050
    assert client.last_body["startDate"] == "2026-06-01T00:00:00+03:00"


def test_fetch_realtime_generation_empty():
    client = FakeClient(generation={"items": []})
    df = fetch_realtime_generation(client, 1, dt.date(2026, 6, 1), dt.date(2026, 6, 2))
    assert df.height == 0
    assert df.columns == ["ts", "power_mw"]


@respx.mock
def test_fetch_aic_parses_items():
    respx.post(CAS_URL).mock(return_value=httpx.Response(201, text="TGT-test"))
    route = respx.post(f"{BASE_URL}/generation/data/aic").mock(
        return_value=httpx.Response(200, json={"body": {"items": [
            {"date": "2026-06-01T00:00:00+03:00", "aic": 41.5},
            {"date": "2026-06-01T01:00:00+03:00", "availableCapacity": "40.25"},
        ]}})
    )
    client = EpiasClient(username="user", password="secret")

    df = fetch_aic(client, 2050, dt.date(2026, 6, 1), dt.date(2026, 6, 2))

    assert df.schema == {"ts": pl.Datetime("us"), "aic_mw": pl.Float64}
    assert df.rows() == [
        (dt.datetime(2026, 5, 31, 21), 41.5),
        (dt.datetime(2026, 5, 31, 22), 40.25),
    ]
    assert route.called
    assert route.calls.last.request.headers["TGT"] == "TGT-test"
    assert json.loads(route.calls.last.request.content) == {
        "startDate": "2026-06-01T00:00:00+03:00",
        "endDate": "2026-06-02T00:00:00+03:00",
        "powerPlantId": 2050,
    }


@respx.mock
def test_fetch_dpp_parses_items():
    respx.post(CAS_URL).mock(return_value=httpx.Response(201, text="TGT-test"))
    route = respx.post(f"{BASE_URL}/generation/data/dpp").mock(
        return_value=httpx.Response(200, json={"items": [
            {"date": "2026-06-01T00:00:00+03:00", "dpp": 35},
            {"dateTime": "2026-06-01T01:00:00+03:00", "productionPlan": 36.75},
        ]})
    )
    client = EpiasClient(username="user", password="secret")

    df = fetch_dpp(client, 2050, dt.date(2026, 6, 1), dt.date(2026, 6, 2))

    assert df.schema == {"ts": pl.Datetime("us"), "dpp_mw": pl.Float64}
    assert df.rows() == [
        (dt.datetime(2026, 5, 31, 21), 35.0),
        (dt.datetime(2026, 5, 31, 22), 36.75),
    ]
    assert route.called


@pytest.mark.parametrize(
    ("fetcher", "path", "item", "message"),
    [
        (fetch_aic, "/generation/data/aic", {"date": "bad", "aic": 1}, "tarih"),
        (
            fetch_dpp,
            "/generation/data/dpp",
            {"date": "2026-06-01T00:00:00+03:00", "dpp": None},
            "null",
        ),
        (
            fetch_aic,
            "/generation/data/aic",
            {"date": "2026-06-01T00:00:00+03:00", "aic": "not-a-number"},
            "sayısal",
        ),
    ],
)
def test_fetch_operator_signal_validation(fetcher, path, item, message):
    client = FakeClient(responses={path: {"items": [item]}})
    with pytest.raises(ValueError, match=message):
        fetcher(client, 1, dt.date(2026, 6, 1), dt.date(2026, 6, 1))
