"""Agent identity uses the existing bounded canonical JSON contract."""

import hashlib
import unicodedata

from ..replay.contracts import canonical_json


def _version_fingerprint(agent_key, params):
    normalized = {}
    if type(params) is not dict or len(params) > 20:
        raise ValueError("Invalid agent parameters")
    for key, value in params.items():
        if type(key) is not str or not 1 <= len(key.encode("utf-16-le")) // 2 <= 100:
            raise ValueError("Invalid agent parameter key")
        key.encode("utf-8")
        if type(value) not in (str, int, float, bool, type(None)):
            raise ValueError("Invalid agent scalar")
        if type(value) is str:
            value.encode("utf-8")
            if len(value.encode("utf-16-le")) // 2 > 2000 or any(
                unicodedata.category(char) == "Cn" for char in value
            ):
                raise ValueError("Unsupported NFC Unicode version")
            value = unicodedata.normalize("NFC", value)
        normalized[key] = value
    return hashlib.sha256(
        canonical_json({"version": 1, "agentKey": agent_key, "params": normalized})
    ).hexdigest()


class AgentCanonicalError(ValueError):
    pass


def version_fingerprint(agent_key, params):
    try:
        return _version_fingerprint(agent_key, params)
    except (ValueError, UnicodeError, OverflowError) as error:
        raise AgentCanonicalError("Unsupported agent canonical identity") from error


def attributes_digest(attributes, keys=None):
    selected = (
        {key: attributes[key] for key in keys if key in attributes}
        if keys is not None
        else dict(attributes)
    )
    return hashlib.sha256(canonical_json({"version": 1, "attributes": selected})).hexdigest()
