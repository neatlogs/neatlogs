"""Refresh ``public-api.json`` from the Neatlogs public OpenAPI document.

    python tests/fixtures/contracts/extract.py <path to public-openapi.json> [source label]

Keeps only the operations neatlogs-harbor calls and the component schemas they
reference, so the tests check every request and response against the real
contract without vendoring the whole document.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OPERATIONS = [
    ("get", "/api/v1/public/projects"),
    ("get", "/api/v1/public/datasets"),
    ("post", "/api/v1/public/datasets"),
    ("get", "/api/v1/public/datasets/{datasetId}/cases"),
    ("post", "/api/v1/public/runs"),
    ("get", "/api/v1/public/runs/{runId}"),
    ("post", "/api/v1/public/runs/{runId}/outputs"),
    ("post", "/api/v1/public/runs/{runId}/complete"),
    ("get", "/api/v1/public/runs/{runId}/comparison"),
    ("post", "/api/v1/public/runs/{runId}/tags"),
    ("get", "/api/v1/public/traces/{traceId}"),
]


def _refs(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/"):
            found.add(ref)
        for item in value.values():
            _refs(item, found)
    elif isinstance(value, list):
        for item in value:
            _refs(item, found)


def extract(document: dict[str, Any], source: str) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for method, path in OPERATIONS:
        paths.setdefault(path, {})[method] = document["paths"][path][method]
    refs: set[str] = set()
    _refs(paths, refs)
    components: dict[str, dict[str, Any]] = {}
    pending = sorted(refs)
    while pending:
        ref = pending.pop()
        _, _, section, name = ref.split("/", 3)
        if name in components.get(section, {}):
            continue
        value = document["components"][section][name]
        components.setdefault(section, {})[name] = value
        nested: set[str] = set()
        _refs(value, nested)
        pending.extend(sorted(nested - refs))
        refs |= nested
    return {
        "openapi": document["openapi"],
        "info": {**document["info"], "x-extracted-from": source},
        "paths": paths,
        "components": components,
    }


if __name__ == "__main__":
    source_path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else source_path.name
    subset = extract(json.loads(source_path.read_text()), label)
    target = Path(__file__).with_name("public-api.json")
    target.write_text(json.dumps(subset, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target} ({len(subset['paths'])} paths)")
