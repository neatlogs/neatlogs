"""Pricing schema, model definitions, and the PricingProvider chain.

Schema (v2):

* ``usage_types`` is an open dict — ``input`` / ``output`` /
  ``cache_read`` / ``cache_write`` / ``reasoning`` / ``image`` /
  ``audio`` / ``embedding`` / ... — USD per 1M tokens. Adding a new
  cost dimension is a JSON edit, not a code change.
* ``capabilities`` is a list of strings — ``vision`` / ``tools`` /
  ``json_mode`` / ``prompt_cache`` / ``reasoning`` / ``audio`` /
  ``image_input`` / ``embedding``. Third-party providers can extend
  without code changes.
* ``tiers`` is per-usage-type (``{input: [...], output: [...]}``). The
  largest crossed ``above_tokens`` threshold wins.
* ``context_window`` is the model's max input tokens.

The chain pattern matches Langfuse's model-definition resolution:
override first, then bundled, then fallback to provider alias / regex /
heuristic. New providers (litellm mirror, HTTP catalog) plug in without
touching the engine.
"""

from __future__ import annotations

import abc
import dataclasses
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class Capability:
    """Standard capability names. The catalog is a free-form set, so
    third-party providers can extend this without code changes —
    compatibility checks only fail closed on explicit declarations."""

    VISION = "vision"
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    PROMPT_CACHE = "prompt_cache"
    REASONING = "reasoning"
    AUDIO = "audio"
    IMAGE_INPUT = "image_input"
    EMBEDDING = "embedding"


class UsageType:
    """Standard usage type names. The ``usage_types`` dict on a
    ``ModelDefinition`` can contain any string key, so this list is
    documentation rather than a closed enum."""

    INPUT = "input"
    OUTPUT = "output"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    REASONING = "reasoning"
    IMAGE = "image"
    AUDIO = "audio"


@dataclasses.dataclass
class Tier:
    """A single tiered rate: if the usage count is strictly above
    ``above_tokens``, use ``rate`` instead of the base rate. A model
    with multiple tiers picks the largest ``above_tokens`` that is
    crossed."""

    above_tokens: int
    rate: float


@dataclasses.dataclass
class ModelDefinition:
    """Pricing and capability data for one model.

    ``usage_types`` maps a usage type to USD per 1M tokens. Missing
    keys mean that usage type is not billed (or not supported).

    ``tiers`` is a per-usage-type list of ``Tier`` entries. When a
    span's token count crosses a tier threshold, that tier's rate
    is used. The largest crossed tier wins.
    """

    model_key: str
    provider: str
    context_window: int | None = None
    capabilities: Set[str] = dataclasses.field(default_factory=set)
    usage_types: Dict[str, float] = dataclasses.field(default_factory=dict)
    tiers: Dict[str, List[Tier]] = dataclasses.field(default_factory=dict)

    def rate_for(self, usage_type: str) -> Optional[float]:
        return self.usage_types.get(usage_type)

    def effective_rate(self, usage_type: str, count: int) -> Optional[float]:
        base = self.rate_for(usage_type)
        if base is None:
            return None
        if count <= 0 or usage_type not in self.tiers:
            return base
        crossed = [t for t in self.tiers[usage_type] if t.above_tokens < count]
        if not crossed:
            return base
        return max(crossed, key=lambda t: t.above_tokens).rate

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def has_all_capabilities(self, caps: Iterable[str]) -> bool:
        return all(c in self.capabilities for c in caps)

    def missing_capabilities(self, caps: Iterable[str]) -> Set[str]:
        return {c for c in caps if c not in self.capabilities}


class PricingProvider(abc.ABC):
    """Source of ``ModelDefinition`` for a given model key.

    Implementations include ``BuiltinProvider`` (bundled JSON),
    ``CustomProvider`` (user-supplied override file), and any future
    HTTP-backed or litellm-mirror provider. ``ChainProvider`` composes
    multiple providers in priority order.
    """

    @abc.abstractmethod
    def lookup(self, model_key: str) -> Optional[ModelDefinition]: ...

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        return None

    def catalog(self) -> List[ModelDefinition]:
        """Return every model known to this provider.

        Default returns an empty list. Concrete providers with a
        static catalog (built-in JSON, custom override file) override
        this. Remote providers that can't enumerate (litellm mirror)
        can leave the default; ``ChainProvider`` will simply skip them.
        """
        return []


def _build_def(key: str, raw: Dict) -> Optional[ModelDefinition]:
    if not isinstance(raw, dict):
        return None
    provider = str(raw.get("provider", "")).lower()
    if "/" not in key or not provider:
        return None
    caps_raw = raw.get("capabilities") or []
    if not isinstance(caps_raw, list):
        caps_raw = []
    capabilities = {str(c) for c in caps_raw if isinstance(c, str)}
    usage_types_raw = raw.get("usage_types") or {}
    if not isinstance(usage_types_raw, dict):
        usage_types_raw = {}
    usage_types = {
        str(k): float(v) for k, v in usage_types_raw.items() if isinstance(v, (int, float))
    }
    tiers_raw = raw.get("tiers") or {}
    if not isinstance(tiers_raw, dict):
        tiers_raw = {}
    tiers: Dict[str, List[Tier]] = {}
    for usage_type, entries in tiers_raw.items():
        if not isinstance(entries, list):
            continue
        out_entries: List[Tier] = []
        for e in entries:
            if (
                isinstance(e, dict)
                and isinstance(e.get("above_tokens"), int)
                and isinstance(e.get("rate"), (int, float))
            ):
                out_entries.append(Tier(above_tokens=e["above_tokens"], rate=float(e["rate"])))
        if out_entries:
            tiers[str(usage_type)] = out_entries
    cw_raw = raw.get("context_window")
    context_window = cw_raw if isinstance(cw_raw, int) and not isinstance(cw_raw, bool) else None
    return ModelDefinition(
        model_key=key,
        provider=provider,
        context_window=context_window,
        capabilities=capabilities,
        usage_types=usage_types,
        tiers=tiers,
    )


class BuiltinProvider(PricingProvider):
    """Reads ``neatlogs/config/pricing.json``."""

    def __init__(self, path: Optional[str | os.PathLike] = None):
        src = (
            Path(path)
            if path is not None
            else Path(__file__).parent.parent / "config" / "pricing.json"
        )
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._models: Dict[str, ModelDefinition] = {}
        self._by_pn: Dict[Tuple[str, str], str] = {}
        for key, raw in (data.get("models") or {}).items():
            d = _build_def(key, raw)
            if d is not None:
                self._models[key] = d
                self._by_pn[(d.provider, key.split("/", 1)[1])] = key

    def lookup(self, model_key: str) -> Optional[ModelDefinition]:
        return self._models.get(model_key)

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        if provider:
            key = self._by_pn.get((provider.lower(), model_name))
            if key:
                return self._models.get(key)
            return None
        for prov, name in self._by_pn.keys():
            if name == model_name:
                return self._models.get(self._by_pn[(prov, name)])
        return None

    def catalog(self) -> List[ModelDefinition]:
        return list(self._models.values())


class CustomProvider(PricingProvider):
    """A user-supplied override file. Sits on top of the builtin catalog
    in a ``ChainProvider`` so the override wins for any model it declares
    and falls through for everything else.
    """

    def __init__(self, path: str | os.PathLike):
        self._path = Path(path)
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._models: Dict[str, ModelDefinition] = {}
        self._by_pn: Dict[Tuple[str, str], str] = {}
        for key, raw in (data.get("models") or {}).items():
            d = _build_def(key, raw)
            if d is not None:
                self._models[key] = d
                self._by_pn[(d.provider, key.split("/", 1)[1])] = key

    def lookup(self, model_key: str) -> Optional[ModelDefinition]:
        return self._models.get(model_key)

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        if provider:
            key = self._by_pn.get((provider.lower(), model_name))
            if key:
                return self._models.get(key)
            return None
        for prov, name in self._by_pn.keys():
            if name == model_name:
                return self._models.get(self._by_pn[(prov, name)])
        return None

    def catalog(self) -> List[ModelDefinition]:
        return list(self._models.values())


class ChainProvider(PricingProvider):
    """Composes a list of providers in priority order. The first one
    to return a non-None result wins. Matches the Langfuse model-
    definition resolution pattern: custom override first, then bundled
    catalog, then fallback to provider alias / regex / heuristic."""

    def __init__(self, providers: Sequence[PricingProvider]):
        self._providers = list(providers)

    def lookup(self, model_key: str) -> Optional[ModelDefinition]:
        for p in self._providers:
            r = p.lookup(model_key)
            if r is not None:
                return r
        return None

    def lookup_by_provider_and_name(
        self, provider: Optional[str], model_name: str
    ) -> Optional[ModelDefinition]:
        for p in self._providers:
            r = p.lookup_by_provider_and_name(provider, model_name)
            if r is not None:
                return r
        return None

    def catalog(self) -> List[ModelDefinition]:
        seen: Set[str] = set()
        out: List[ModelDefinition] = []
        for p in self._providers:
            for m in p.catalog():
                if m.model_key in seen:
                    continue
                seen.add(m.model_key)
                out.append(m)
        return out


def default_chain(pricing_file: Optional[str | os.PathLike] = None) -> ChainProvider:
    """Build the default chain: optional user override, then bundled."""
    providers: List[PricingProvider] = []
    if pricing_file is not None:
        providers.append(CustomProvider(pricing_file))
    elif os.environ.get("NEATLOGS_PRICING_FILE"):
        providers.append(CustomProvider(os.environ["NEATLOGS_PRICING_FILE"]))
    providers.append(BuiltinProvider())
    return ChainProvider(providers)
