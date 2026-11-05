"""Short-lived CLI access-token renewal for one selected task host and project."""

import json
import os
import re
import threading
from urllib.parse import urlsplit

import httpx
from opentelemetry.instrumentation.utils import suppress_instrumentation


def task_credential_refresh(*, token, project_id, base_url):
    endpoint = os.getenv("NEATLOGS_CREDENTIAL_BRIDGE_URL")
    key = os.getenv("NEATLOGS_CREDENTIAL_BRIDGE_KEY")
    if (
        not endpoint
        or not key
        or token != os.getenv("NEATLOGS_TOKEN")
        or project_id != os.getenv("NEATLOGS_PROJECT_ID")
        or base_url.rstrip("/") != os.getenv("NEATLOGS_HOST")
    ):
        return None
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.path != "/refresh"
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        raise ValueError("Invalid CLI credential bridge")
    lock = threading.Lock()
    current = token

    def refresh(rejected):
        nonlocal current
        with lock:
            if current != rejected:
                return current
            try:
                with (
                    suppress_instrumentation(),
                    httpx.stream(
                        "POST",
                        endpoint,
                        headers={"Authorization": "Bearer " + key},
                        json={"rejectedToken": rejected},
                        timeout=5,
                        follow_redirects=False,
                        trust_env=False,
                    ) as response,
                ):
                    if response.status_code != 200:
                        raise ValueError()
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > 1024:
                            raise ValueError()
                    result = json.loads(data)
                    value = result.get("token")
                    if (
                        not isinstance(value, str)
                        or not re.fullmatch(r"nloa_[A-Za-z0-9_-]{43}", value)
                        or set(result) != {"token"}
                    ):
                        raise ValueError()
                    current = value
                    return current
            except Exception:
                raise RuntimeError(
                    "CLI sign-in could not be renewed. Sign in again and resume the saved run."
                ) from None

    return refresh
