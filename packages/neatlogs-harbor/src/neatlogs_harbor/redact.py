"""Client-side redaction applied to everything before it leaves the machine.

Harbor already scrubs the secrets it knows about from trial files. This adds two
more layers before upload:

* token-shaped strings (provider API keys, cloud keys, JWTs, bearer tokens,
  private key blocks, ``password=...`` assignments), matched by pattern; and
* the exact values of secret-looking environment variables in the Harbor
  process (``*_API_KEY``, ``*_TOKEN``, ``*_SECRET``, ``*_PASSWORD``, ``*_KEY``),
  which catches keys whose shape no pattern knows.

Extra patterns come from ``--pk redact=<regex>`` (several separated by ``;;``)
or ``NEATLOGS_HARBOR_REDACT``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "[REDACTED]"

_TOKEN_PATTERNS: tuple[str, ...] = (
    # Private key blocks.
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    # Provider API keys: OpenAI (sk-, sk-proj-), Anthropic (sk-ant-), OpenRouter (sk-or-).
    r"\bsk-(?:ant-|proj-|or-)?[A-Za-z0-9_-]{20,}",
    # Neatlogs keys.
    r"\bnl(?:pk)?_[A-Za-z0-9_-]{16,}",
    # GitHub tokens.
    r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    # AWS access key ids.
    r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b",
    # Google API keys.
    r"\bAIza[0-9A-Za-z_-]{35}",
    # Slack tokens.
    r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    # Stripe keys.
    r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}",
    # Hugging Face tokens.
    r"\bhf_[A-Za-z0-9]{30,}",
    # JSON Web Tokens.
    r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    # Bearer credentials in headers or logs.
    r"(?i)(?<=bearer )[A-Za-z0-9._~+/-]{20,}=*",
)

# ``api_key = "..."``, ``"token": "..."``, ``PASSWORD=...``: keep the name, drop the value.
_ASSIGNMENT = re.compile(
    r"(?i)(?P<name>(?<![A-Za-z0-9])(?:api[_-]?key|secret|token|password|passwd)"
    r"[\"']?\s*[:=]\s*[\"']?)(?P<value>[^\s\"',;}]{8,})"
)

_SECRET_ENV_NAME = re.compile(r"(?i)(?:^|_)(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|KEY)$")
_MIN_SECRET_VALUE_LEN = 12


class Redactor:
    """Replaces secrets in strings and in nested JSON-like values."""

    def __init__(
        self,
        extra_patterns: Iterable[str] = (),
        *,
        secret_values: Iterable[str] = (),
    ) -> None:
        self._patterns = [re.compile(p) for p in (*_TOKEN_PATTERNS, *extra_patterns) if p]
        values = {v for v in secret_values if v and len(v) >= _MIN_SECRET_VALUE_LEN}
        # Longest first so a value that contains another is replaced whole.
        self._values = sorted(values, key=len, reverse=True)

    @classmethod
    def from_environment(
        cls,
        extra_patterns: Iterable[str] = (),
        environ: Mapping[str, str] | None = None,
        *,
        extra_secrets: Iterable[str] = (),
    ) -> Redactor:
        env = os.environ if environ is None else environ
        secrets = [value for name, value in env.items() if _SECRET_ENV_NAME.search(name)]
        return cls(extra_patterns, secret_values=[*secrets, *extra_secrets])

    def text(self, value: str) -> str:
        if not value:
            return value
        for secret in self._values:
            if secret in value:
                value = value.replace(secret, REDACTED)
        for pattern in self._patterns:
            value = pattern.sub(REDACTED, value)
        return _ASSIGNMENT.sub(lambda m: f"{m.group('name')}{REDACTED}", value)

    def value(self, value: Any) -> Any:
        """Redact every string inside a JSON-like value (dicts, lists, scalars)."""

        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        return value


def split_patterns(raw: str | None) -> list[str]:
    """Parse the ``redact`` option: several regexes separated by ``;;``."""

    if not raw:
        return []
    return [part.strip() for part in raw.split(";;") if part.strip()]
