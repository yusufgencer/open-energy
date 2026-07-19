import httpx
import pytest
import respx

from openenergy.providers.base import ApiRole
from openenergy.providers.openmeteo.client import (
    BASE_URLS,
    FixedWindowRateLimiter,
    OpenMeteoClient,
)


@respx.mock
def test_get_returns_json():
    route = respx.get(BASE_URLS[ApiRole.FORECAST]).mock(
        return_value=httpx.Response(200, json={"hourly": {"time": []}})
    )
    client = OpenMeteoClient()
    data = client.get(ApiRole.FORECAST, {"latitude": 39.9, "longitude": 32.8})
    assert route.called
    assert data == {"hourly": {"time": []}}


@respx.mock
def test_retries_on_429_then_succeeds():
    route = respx.get(BASE_URLS[ApiRole.FORECAST]).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = OpenMeteoClient(max_retries=3, backoff_base=0.0)
    data = client.get(ApiRole.FORECAST, {"latitude": 1, "longitude": 2})
    assert data == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_backoff_has_full_jitter():
    route = respx.get(BASE_URLS[ApiRole.FORECAST]).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    random_values = iter([0.25, 0.75])
    delays: list[float] = []
    client = OpenMeteoClient(
        max_retries=2,
        backoff_base=2.0,
        rng=lambda: next(random_values),
        sleep=delays.append,
    )

    assert client.get(ApiRole.FORECAST, {"latitude": 1, "longitude": 2}) == {"ok": True}
    assert route.call_count == 3
    assert delays == [0.5, 3.0]


@respx.mock
def test_retry_after_is_a_lower_bound_for_jitter():
    respx.get(BASE_URLS[ApiRole.FORECAST]).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    delays: list[float] = []
    client = OpenMeteoClient(
        max_retries=1,
        backoff_base=2.0,
        rng=lambda: 0.5,
        sleep=delays.append,
    )

    client.get(ApiRole.FORECAST, {"latitude": 1, "longitude": 2})
    assert delays == [7.0]


def test_rate_limiter_accounts_for_batch_weight():
    current = [0.0]
    waits: list[float] = []

    def sleep(delay: float) -> None:
        waits.append(delay)
        current[0] += delay

    limiter = FixedWindowRateLimiter(
        limit=5,
        window_seconds=10,
        clock=lambda: current[0],
        sleep=sleep,
    )
    limiter.acquire(weight=4)
    limiter.acquire(weight=1)
    limiter.acquire(weight=2)

    assert waits == [10.0]


@respx.mock
def test_raises_after_exhausting_retries():
    respx.get(BASE_URLS[ApiRole.FORECAST]).mock(return_value=httpx.Response(429))
    client = OpenMeteoClient(max_retries=2, backoff_base=0.0)
    with pytest.raises(httpx.HTTPStatusError):
        client.get(ApiRole.FORECAST, {"latitude": 1, "longitude": 2})


@respx.mock
def test_api_key_uses_customer_host_and_appends_key():
    from openenergy.providers.openmeteo.client import CUSTOMER_URLS

    route = respx.get(CUSTOMER_URLS[ApiRole.FORECAST]).mock(
        return_value=httpx.Response(200, json={"ok": 1})
    )
    client = OpenMeteoClient(api_key="secret")
    client.get(ApiRole.FORECAST, {"latitude": 1, "longitude": 2})
    assert route.called
    assert route.calls.last.request.url.params["apikey"] == "secret"


@respx.mock
def test_single_runs_role_routes_to_customer_previous_runs_host():
    # Single Runs (run= param) is served by the previous-runs host; the client
    # must expose SINGLE_RUNS as its own routable role (paid → customer host).
    from openenergy.providers.openmeteo.client import CUSTOMER_URLS

    route = respx.get(CUSTOMER_URLS[ApiRole.SINGLE_RUNS]).mock(
        return_value=httpx.Response(200, json={"ok": 1})
    )
    client = OpenMeteoClient(api_key="secret")
    client.get(ApiRole.SINGLE_RUNS, {"latitude": 1, "longitude": 2, "run": 123})
    assert route.called
    assert route.calls.last.request.url.params["apikey"] == "secret"
