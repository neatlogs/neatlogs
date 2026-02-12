"""
Model defaults enricher for invocation parameters.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DefaultsEnricher:
    """
    Enriches invocation parameters with model-specific defaults.

    Loads defaults from config/model_defaults.json and intelligently merges
    them with explicitly set parameters.
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
            except (FileNotFoundError, json.JSONDecodeError) as e:
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

        provider_data = defaults_data.get(provider.lower(), {})
        operation_data = provider_data.get(operation, {})

        if not operation_data:
            return {}

        if model in operation_data:
            return operation_data[model].copy()

        for model_key, defaults in operation_data.items():
            if model_key != "_default" and model.startswith(model_key):
                logger.debug(f"Matched model '{model}' to defaults for '{model_key}'")
                return defaults.copy()

        if "_default" in operation_data:
            logger.debug(f"Using _default for {provider}/{operation}/{model}")
            return operation_data["_default"].copy()

        return {}


def enrich_invocation_parameters(
    merged_attrs: Dict[str, Any], enable_enrichment: bool = True
) -> None:
    """
    Enrich invocation parameters with model defaults.

    Merges default parameters from model_defaults.json with explicitly captured
    parameters. Explicit parameters always take precedence.

    Modifies merged_attrs in-place by adding/updating:
    - llm.invocation_parameters.full: Complete params (explicit + defaults)
    - llm.invocation_parameters.enriched: Flag indicating enrichment was applied

    Args:
        merged_attrs: Span attributes dictionary
        enable_enrichment: Whether to apply enrichment
    """
    if not enable_enrichment:
        return

    span_kind = merged_attrs.get("openinference.span.kind")
    if span_kind not in ("LLM", "EMBEDDING"):
        return

    provider = merged_attrs.get("llm.system", "").lower()
    # For EMBEDDING spans, use embedding.model_name; for LLM spans, use llm.model_name
    if span_kind == "EMBEDDING":
        model = merged_attrs.get("embedding.model_name", "")
    else:
        model = merged_attrs.get("llm.model_name", "")

    if not provider or not model:
        return

    operation = None
    if span_kind == "LLM":
        if "chat" in merged_attrs.get("llm.request.type", "").lower():
            operation = "chat.completions" if provider == "openai" else "messages"
        else:
            operation = "chat.completions" if provider == "openai" else "messages"
    elif span_kind == "EMBEDDING":
        operation = "embeddings"

    if not operation:
        return

    defaults = DefaultsEnricher.get_defaults(provider, operation, model)
    if not defaults:
        logger.debug(f"No defaults found for {provider}/{operation}/{model}")
        return

    existing_params_str = merged_attrs.get("llm.invocation_parameters", "{}")
    try:
        existing_params = (
            json.loads(existing_params_str)
            if isinstance(existing_params_str, str)
            else existing_params_str
        )
    except json.JSONDecodeError:
        existing_params = {}

    enriched_params = {**defaults, **existing_params}

    merged_attrs["llm.invocation_parameters"] = json.dumps(enriched_params)

    logger.debug(f"✓ Enriched params for {provider}/{model}: added {len(defaults)} defaults")
