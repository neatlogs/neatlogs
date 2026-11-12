"""Checks requests and responses against the vendored public API contract.

``fixtures/contracts/public-api.json`` is the part of Neatlogs' public OpenAPI
document (generated from the backend's zod schemas) that neatlogs-harbor uses;
``fixtures/contracts/extract.py`` refreshes it.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "contracts" / "public-api.json"
_ROOT = "urn:neatlogs-public-api"


@cache
def _document() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


@cache
def _registry() -> Registry:
    resource = Resource.from_contents(_document(), default_specification=DRAFT202012)
    return Registry().with_resource(_ROOT, resource)


def _validator(schema: dict[str, Any]) -> Draft202012Validator:
    rooted = json.loads(json.dumps(schema).replace('"#/components/', f'"{_ROOT}#/components/'))
    return Draft202012Validator(rooted, registry=_registry())


def _errors(schema: dict[str, Any], value: Any, where: str) -> list[str]:
    return [
        f"{where}: {error.message} at {'/'.join(map(str, error.absolute_path)) or '<root>'}"
        for error in _validator(schema).iter_errors(value)
    ]


def _resolve(value: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in value:
        _, _, section, name = value["$ref"].split("/", 3)
        value = _document()["components"][section][name]
    return value


def operation(method: str, path: str) -> tuple[str, dict[str, Any]] | None:
    for template, operations in _document()["paths"].items():
        pattern = "^" + re.sub(r"\{[^}]+\}", "[^/]+", template) + "$"
        if re.match(pattern, path) and method.lower() in operations:
            return template, operations[method.lower()]
    return None


def _coerce(raw: str, schema: dict[str, Any]) -> Any:
    kinds = {schema.get("type")} | {s.get("type") for s in schema.get("anyOf", [])}
    if "integer" in kinds:
        try:
            return int(raw)
        except ValueError:
            return raw
    if "number" in kinds:
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def check_request(request: httpx.Request) -> list[str]:
    found = operation(request.method, request.url.path)
    if found is None:
        return [f"{request.method} {request.url.path} is not in the contract"]
    template, op = found
    where = f"{request.method} {template}"
    problems: list[str] = []
    for parameter in op.get("parameters", []):
        parameter = _resolve(parameter)
        name, location = parameter["name"], parameter["in"]
        schema = parameter.get("schema", {})
        if location == "header":
            value = request.headers.get(name)
        elif location == "query":
            raw = request.url.params.get(name)
            value = None if raw is None else _coerce(raw, _resolve(schema))
        else:
            continue
        if value is None:
            if parameter.get("required"):
                problems.append(f"{where}: missing required {location} {name}")
            continue
        problems += _errors(schema, value, f"{where} {location} {name}")
    body_spec = op.get("requestBody")
    if body_spec is not None:
        schema = body_spec["content"]["application/json"]["schema"]
        if not request.content:
            if body_spec.get("required"):
                problems.append(f"{where}: missing request body")
        else:
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                problems.append(f"{where}: body must be application/json")
            problems += _errors(schema, json.loads(request.content), f"{where} body")
    return problems


def check_response(method: str, path: str, status: int, payload: Any) -> list[str]:
    found = operation(method, path)
    if found is None or status >= 300:
        return []
    template, op = found
    response = op["responses"].get(str(status))
    if response is None:
        return [f"{method} {template}: status {status} is not documented"]
    response = _resolve(response)
    schema = (response.get("content") or {}).get("application/json", {}).get("schema")
    if schema is None:
        return []
    return _errors(schema, payload, f"{method} {template} {status} response")
