"""OTLP HTTP session policy that complements (without duplicating) OTel retry."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_otlp_session() -> requests.Session:
    """Retry POST 429/read-timeout failures that OTel Python 1.43 does not.

    The upstream exporter already retries 408, 5xx, and connection failures.
    Restricting this adapter to 429 and read errors avoids nested retries for
    the statuses handled by that outer OTLP loop.
    """
    retry = Retry(
        total=2,
        connect=0,
        read=2,
        status=2,
        other=0,
        allowed_methods=frozenset({"POST"}),
        status_forcelist=frozenset({429}),
        backoff_factor=0.1,
        respect_retry_after_header=False,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
