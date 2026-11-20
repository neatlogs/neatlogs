"""
Model defaults enricher for invocation parameters.
"""

import json
import logging
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import for provider detection
try:
    from ..core.instrumentation_scope_parser import get_effective_provider_for_defaults
except ImportError:
    # Fallback if import fails
    def get_effective_provider_for_defaults(attrs: Dict[str, Any]) -> str:
        return attrs.get("llm.system", "").lower()


# Operation keys declared by model_defaults.json. The mapping is intentionally
# explicit: assuming "messages" for every non-OpenAI provider means Google,
# Vertex AI and Cohere spans never find their defaults at all.
_CHAT_OPERATION_BY_PROVIDER: Dict[str, str] = {
    "openai": "chat.completions",
    "azure": "chat.completions",
    "azure_openai": "chat.completions",
    "groq": "chat.completions",
    "xai": "chat.completions",
    "deepseek": "chat.completions",
    "mistral": "chat",
    "mistralai": "chat",
    "ollama": "chat",
    "anthropic": "messages",
    "bedrock": "messages",
    "google": "generateContent",
    "gemini": "generateContent",
    "vertex_ai": "generateContent",
    "vertexai": "generateContent",
    "cohere": "chat",
}

# Canonical spellings for explicit operation names emitted by instrumentors.
_EXPLICIT_OPERATIONS: Dict[str, str] = {
    "chat": "chat",
    "chat.completions": "chat.completions",
    "chat_completions": "chat.completions",
    "messages": "messages",
    "generatecontent": "generateContent",
    "generate_content": "generateContent",
    "generate": "generate",
    "embeddings": "embeddings",
    "embedding": "embeddings",
}

_EMBEDDING_OPERATION = "embeddings"


def resolve_default_operations(
    provider: str, span_kind: str, attrs: Mapping[str, Any]
) -> List[str]:
    """Return candidate ``model_defaults.json`` operation keys in priority order.

    An explicit operation reported by the instrumentor is tried first, but only
    after it has been canonicalized to a key the defaults file can actually use.
    The provider's conventional chat operation is always appended as a fallback,
    so a misleading ``llm.request.type`` cannot suppress enrichment entirely.
    """

    if span_kind == "EMBEDDING":
        return [_EMBEDDING_OPERATION]

    candidates: List[str] = []
    for key in ("gen_ai.operation.name", "llm.request.type"):
        value = attrs.get(key)
        if not isinstance(value, str):
            continue
        operation = _EXPLICIT_OPERATIONS.get(value.strip().lower())
        if operation and operation not in candidates:
            candidates.append(operation)

    provider_key = str(provider or "").strip().lower()
    fallback = _CHAT_OPERATION_BY_PROVIDER.get(provider_key, "chat.completions")
    if fallback not in candidates:
        candidates.append(fallback)
    return candidates


class DefaultsEnricher:
    """
    Enriches invocation parameters with model-specific defaults.

    Loads defaults from config/model_defaults.json and intelligently merges
    them with explicitly captured parameters. Explicit parameters always take
    precedence.
    """

    _defaults_data: Optional[Dict[str, Any]] = None

    @classmethod
    def _load_defaults(cls) -> Dict[str, Any]:
        """Load model defaults from JSON file."""
        if cls._defaults_data is None:
            defaults_file = os.path.join(os.path.dirname(__file__), "model_defaults.json")
            try:
                with open(defaults_file, "r") as f:
                    cls._defaults_data = json.load(f)
                logger.debug(f"✓ Loaded model defaults from {defaults_file}")
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load model_defaults.json: {e}")
                cls._defaults_data = {}
        return cls._defaults_data

    @classmethod
    def get_defaults(cls, provider: str, operation: str, model: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific provider/operation/model.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            operation: Operation type (e.g., "chat.completions", "messages")
            model: Model name (e.g., "gpt-4", "claude-3-5-sonnet")

        Returns:
            Dictionary of default parameters
        """

        defaults_data = cls._load_defaults()
        provider_data = defaults_data.get(str(provider or "").strip().lower(), {})
        if not isinstance(provider_data, Mapping):
            return {}
        operation_data = provider_data.get(operation, {})
        if not isinstance(operation_data, Mapping) or not operation_data:
            return {}

        model_name = str(model or "").strip()
        if not model_name:
            return {}
        lowered = model_name.lower()

        # An exact match always wins, regardless of the casing used by the caller.
        for model_key, defaults in operation_data.items():
            key = str(model_key)
            if key != "_default" and key.lower() == lowered and isinstance(defaults, Mapping):
                logger.debug(f"Matched model '{model}' exactly to defaults for '{key}'")
                return dict(defaults)

        # Several configured keys can be prefixes of the same model id
        # (e.g. "gpt-5" and "gpt-5-mini"). Return the longest match so the most
        # specific family wins instead of whichever key was listed first.
        matches = [
            (len(str(model_key)), str(model_key), defaults)
            for model_key, defaults in operation_data.items()
            if model_key != "_default"
            and isinstance(defaults, Mapping)
            and lowered.startswith(str(model_key).lower())
        ]
        if matches:
            matches.sort(key=lambda match: (-match[0], match[1]))
            _, model_key, defaults = matches[0]
            logger.debug(f"Matched model '{model}' to defaults for '{model_key}'")
            return dict(defaults)

        fallback = operation_data.get("_default")
        if isinstance(fallback, Mapping):
            logger.debug(f"Using _default for {provider}/{operation}/{model}")
            return dict(fallback)

        return {}


def enrich_invocation_parameters(
    merged_attrs: Dict[str, Any], enable_enrichment: bool = True
) -> None:
    """
    Enrich invocation parameters with model defaults.

    Merges default parameters from model_defaults.json with explicitly captured
    parameters. Explicit parameters always take precedence.

    Modifies merged_attrs in-place by replacing:
    - llm.invocation_parameters: explicit parameters plus any applicable defaults
      that the caller did not set explicitly.

    Args:
        merged_attrs: Span attributes dictionary
        enable_enrichment: Whether to apply enrichment
    """
    if not enable_enrichment:
        return

    span_kind = str(merged_attrs.get("openinference.span.kind") or "").upper()
    if span_kind not in ("LLM", "EMBEDDING"):
        return

    provider = get_effective_provider_for_defaults(merged_attrs)
    if span_kind == "EMBEDDING":
        model = merged_attrs.get("embedding.model_name", "")
    else:
        model = merged_attrs.get("llm.model_name", "")

    if not provider or not model:
        return

    defaults: Dict[str, Any] = {}
    for operation in resolve_default_operations(provider, span_kind, merged_attrs):
        defaults = DefaultsEnricher.get_defaults(provider, operation, model)
        if defaults:
            break
    if not defaults:
        logger.debug(f"No defaults found for {provider}/{model}")
        return

    existing_params = merged_attrs.get("llm.invocation_parameters", "{}")
    if isinstance(existing_params, str):
        try:
            existing_params = json.loads(existing_params)
        except (TypeError, ValueError):
            existing_params = {}
    if not isinstance(existing_params, Mapping):
        existing_params = {}

    enriched_params = {**defaults, **dict(existing_params)}
    try:
        merged_attrs["llm.invocation_parameters"] = json.dumps(enriched_params)
    except (TypeError, ValueError):
        # Never let a non-serializable caller value break the processing pipeline.
        logger.debug("Could not serialize enriched invocation parameters")
        return

    logger.debug(f"✓ Enriched params for {provider}/{model}: added {len(defaults)} defaults")
