from contextlib import contextmanager
from uuid import uuid4

import httpx
import pytest

from neatlogs._task_credentials import task_credential_refresh
from neatlogs.datasets._api import DatasetApi, DatasetError
from neatlogs.replay.api import LocalRunApi, LocalRunApiError

PROJECT, RUN, CASE = [str(uuid4()) for _ in range(3)]
HOST = "http://127.0.0.1:34567"
TOKEN = "nloa_" + "n" * 43


@pytest.mark.parametrize("kind", ["dataset", "run"])
@pytest.mark.parametrize("status", [200, 401, 403])
def test_task_renewal_is_bounded_and_keeps_request_identity(monkeypatch, kind, status):
    calls, refreshes = [], []
    monkeypatch.setenv("NEATLOGS_TOKEN", "old-token")
    monkeypatch.setenv("NEATLOGS_HOST", HOST)
    monkeypatch.setenv("NEATLOGS_PROJECT_ID", PROJECT)
    monkeypatch.setenv("NEATLOGS_CREDENTIAL_BRIDGE_URL", HOST + "/refresh")
    monkeypatch.setenv("NEATLOGS_CREDENTIAL_BRIDGE_KEY", "private-capability")

    def request(method, url, **kwargs):
        if url.endswith("/refresh"):
            assert kwargs["headers"]["Authorization"] == "Bearer private-capability"
            refreshes.append(kwargs["json"])
            return httpx.Response(200, json={"token": TOKEN})
        calls.append((method, url, dict(kwargs["headers"]), kwargs.get("content")))
        code = status if status != 200 else (200 if kwargs["headers"]["Authorization"] == "Bearer " + TOKEN else 401)
        data = {"outputId": str(uuid4()), "state": "succeeded"}
        return httpx.Response(code, json={"success": code == 200, "data": data})

    @contextmanager
    def stream(*args, **kwargs):
        yield request(*args, **kwargs)

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(httpx, "request", request)
    config = dict(project_id=PROJECT, token="old-token", base_url=HOST)
    config["refresh_access_token"] = task_credential_refresh(**config)
    api = DatasetApi(**config) if kind == "dataset" else LocalRunApi(**config)
    key = str(uuid4())
    run = lambda: api.request("POST", "/datasets", content=b'{}', key=key) if kind == "dataset" else api.request("POST", f"/runs/{RUN}/outputs", {"caseId": CASE, "trial": 1, "output": "saved"}, key=key)
    if status == 200:
        run()
    else:
        with pytest.raises((DatasetError, LocalRunApiError)):
            run()
    assert len(refreshes) == (0 if status == 403 else 1)
    assert len(calls) == (1 if status == 403 else 2)
    if len(calls) == 2:
        assert calls[0][0:2] == calls[1][0:2]
        assert calls[0][3] == calls[1][3]
        assert calls[0][2]["Idempotency-Key"] == calls[1][2]["Idempotency-Key"]


def test_bridge_cannot_renew_for_other_host_or_project(monkeypatch):
    monkeypatch.setenv("NEATLOGS_TOKEN", "old-token")
    monkeypatch.setenv("NEATLOGS_HOST", HOST)
    monkeypatch.setenv("NEATLOGS_PROJECT_ID", PROJECT)
    monkeypatch.setenv("NEATLOGS_CREDENTIAL_BRIDGE_URL", HOST + "/refresh")
    monkeypatch.setenv("NEATLOGS_CREDENTIAL_BRIDGE_KEY", "private-capability")
    assert task_credential_refresh(token="old-token", project_id=str(uuid4()), base_url=HOST) is None
    assert task_credential_refresh(token="old-token", project_id=PROJECT, base_url="https://other.example") is None
    monkeypatch.setenv("NEATLOGS_CREDENTIAL_BRIDGE_URL", "https://external.example/refresh")
    with pytest.raises(ValueError, match="Invalid CLI credential bridge"):
        task_credential_refresh(token="old-token", project_id=PROJECT, base_url=HOST)


def test_delayed_concurrent_rejection_reuses_already_refreshed_token(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    for name, value in {"NEATLOGS_TOKEN": "old-token", "NEATLOGS_HOST": HOST, "NEATLOGS_PROJECT_ID": PROJECT,
        "NEATLOGS_CREDENTIAL_BRIDGE_URL": HOST + "/refresh", "NEATLOGS_CREDENTIAL_BRIDGE_KEY": "private"}.items():
        monkeypatch.setenv(name, value)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    old_requests, renewals = [], []

    @contextmanager
    def stream(*args, **kwargs):
        renewals.append(True)
        yield httpx.Response(200, json={"token": TOKEN})

    def request(*args, **kwargs):
        if kwargs["headers"]["Authorization"] == "Bearer old-token":
            with lock:
                old_requests.append(True)
                number = len(old_requests)
            barrier.wait(timeout=2)
            if number == 2:
                time.sleep(0.1)
            return httpx.Response(401, json={})
        return httpx.Response(200, json={"success": True, "data": {}})

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(httpx, "request", request)
    config = dict(project_id=PROJECT, token="old-token", base_url=HOST)
    api = DatasetApi(**config, refresh_access_token=task_credential_refresh(**config))
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: api.request("GET", "/datasets"), range(2))) == [{}, {}]
    assert len(renewals) == 1
