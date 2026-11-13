"""Settings, all optional: plugin kwargs (``--pk key=value``) win over env vars.

Environment names match the Neatlogs Python SDK. Credentials are read from the
Harbor process, never from the sandbox: Harbor does not forward them to agents.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .redact import split_patterns

DEFAULT_HOST = "https://app.neatlogs.com"
DEFAULT_INGEST = "https://ingest.neatlogs.com"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _number(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def otlp_traces_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"


@dataclass
class Options:
    # Project ingest key (as for the SDK): sends traces to OTLP ingest.
    api_key: str | None = None
    ingest_endpoint: str = DEFAULT_INGEST
    # Public API token (service account) and project: datasets and runs.
    token: str | None = None
    project_id: str | None = None
    host: str = DEFAULT_HOST
    # Runs are graded by a saved evaluation form; required to register runs.
    form_version_id: str | None = None
    evaluator_ids: list[str] = field(default_factory=list)
    # Run name override; defaults to "<agent> · <model>".
    name: str | None = None
    # Comparison baseline (run:<id>, tag:<name>, eval:<id>); defaults to the
    # previous Harbor run on the same dataset version.
    baseline: str | None = None
    redact: list[str] = field(default_factory=list)
    # "default" sends verifier summaries with each output; "none" sends none.
    attachments: str = "default"
    traces: bool = True
    timeout: float = 30.0
    retries: int = 4

    @property
    def otlp_url(self) -> str:
        return otlp_traces_url(self.ingest_endpoint)

    @property
    def grading_configured(self) -> bool:
        return bool(self.form_version_id and self.evaluator_ids)

    @classmethod
    def from_kwargs(
        cls, kwargs: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None
    ) -> Options:
        kw = dict(kwargs or {})
        env = os.environ if environ is None else environ

        def pick(key: str, *env_names: str) -> Any:
            if kw.get(key) not in (None, ""):
                return kw[key]
            for name in env_names:
                if env.get(name):
                    return env[name]
            return None

        return cls(
            api_key=pick("api_key", "NEATLOGS_API_KEY"),
            ingest_endpoint=str(pick("endpoint", "NEATLOGS_ENDPOINT") or DEFAULT_INGEST),
            token=pick("token", "NEATLOGS_TOKEN"),
            project_id=pick("project_id", "NEATLOGS_PROJECT_ID"),
            host=str(pick("host", "NEATLOGS_HOST") or DEFAULT_HOST).rstrip("/"),
            form_version_id=pick("form_version_id", "NEATLOGS_HARBOR_FORM_VERSION_ID"),
            evaluator_ids=_as_list(pick("evaluator_ids", "NEATLOGS_HARBOR_EVALUATOR_IDS")),
            name=pick("name", "NEATLOGS_HARBOR_RUN_NAME"),
            baseline=pick("baseline", "NEATLOGS_HARBOR_BASELINE"),
            redact=split_patterns(pick("redact", "NEATLOGS_HARBOR_REDACT")),
            attachments=str(pick("attachments", "NEATLOGS_HARBOR_ATTACHMENTS") or "default"),
            traces=_as_bool(pick("traces", "NEATLOGS_HARBOR_TRACES"), True),
            timeout=_number(pick("timeout", "NEATLOGS_HARBOR_TIMEOUT"), 30.0),
            retries=int(_number(pick("retries", "NEATLOGS_HARBOR_RETRIES"), 4)),
        )
