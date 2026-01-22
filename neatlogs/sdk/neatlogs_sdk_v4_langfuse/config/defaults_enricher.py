"""
Model defaults enricher for invocation parameters.

Loads default parameter values from model_defaults.json and merges them with
explicitly captured parameters. This helps with debugging by showing both
explicit and default values.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DefaultsEnricher:
    """
    Enriches invocation parameters with model-specific defaults.
    
    Loads defaults from config/model_defaults.json and intelligently merges
    them with explicitly set parameters (explicit params always take precedence).
    """
    
    _defaults_data: Optional[Dict[str, Any]] = None
    
    @classmethod
    def _load_defaults(cls) -> Dict[str, Any]:
        """Load model defaults from JSON file."""
        if cls._defaults_data is None:
            defaults_file = os.path.join(
                os.path.dirname(__file__),
                "model_defaults.json"
            )
            try:
                with open(defaults_file, 'r') as f:
                    cls._defaults_data = json.load(f)
                logger.debug(f"✓ Loaded model defaults from {defaults_file}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load model_defaults.json: {e}")
                cls._defaults_data = {}
        return cls._defaults_data
    
    @classmethod
    def get_defaults(
        cls,
        provider: str,
        operation: str,
        model: str
    ) -> Dict[str, Any]:
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
        
        # Navigate to provider → operation
        provider_data = defaults_data.get(provider.lower(), {})
        operation_data = provider_data.get(operation, {})
        
        if not operation_data:
            return {}
        
        # Try exact model match
        if model in operation_data:
            return operation_data[model].copy()
        
        # Try partial model match (e.g., "gpt-4o-mini-2024" matches "gpt-4o-mini")
        for model_key, defaults in operation_data.items():
            if model_key != "_default" and model.startswith(model_key):
                logger.debug(f"Matched model '{model}' to defaults for '{model_key}'")
                return defaults.copy()
        
        # Fall back to _default
        if "_default" in operation_data:
            logger.debug(f"Using _default for {provider}/{operation}/{model}")
            return operation_data["_default"].copy()
        
        return {}


def enrich_invocation_parameters(
    merged_attrs: Dict[str, Any],
    enable_enrichment: bool = True
) -> None:
    """
    Enrich invocation parameters with model defaults.
    
    Merges default parameters from model_defaults.json with explicitly captured
    parameters. Explicit parameters always take precedence.
    
    Modifies merged_attrs in-place by adding/updating:
    - llm.invocation_parameters.full: Complete params (explicit + defaults)
    - llm.invocation_parameters.enriched: Flag indicating enrichment was applied
    
    Args:
        merged_attrs: Span attributes dictionary (modified in-place)
        enable_enrichment: Whether to apply enrichment (default True)
    """
    if not enable_enrichment:
        return
    
    # Check if this is an LLM span
    span_kind = merged_attrs.get("openinference.span.kind")
    if span_kind not in ("LLM", "EMBEDDING"):
        return
    
    # Get provider and model
    provider = merged_attrs.get("llm.system", "").lower()
    model = merged_attrs.get("llm.model_name", "")
    
    if not provider or not model:
        return
    
    # Determine operation type
    operation = None
    if span_kind == "LLM":
        # Check if it's chat or completion
        if "chat" in merged_attrs.get("llm.request.type", "").lower():
            operation = "chat.completions" if provider == "openai" else "messages"
        else:
            operation = "chat.completions" if provider == "openai" else "messages"
    elif span_kind == "EMBEDDING":
        operation = "embeddings"
    
    if not operation:
        return
    
    # Get defaults for this model
    defaults = DefaultsEnricher.get_defaults(provider, operation, model)
    if not defaults:
        logger.debug(f"No defaults found for {provider}/{operation}/{model}")
        return
    
    # Get existing invocation parameters
    existing_params_str = merged_attrs.get("llm.invocation_parameters", "{}")
    try:
        existing_params = json.loads(existing_params_str) if isinstance(existing_params_str, str) else existing_params_str
    except json.JSONDecodeError:
        existing_params = {}
    
    # Merge: defaults first, then explicit params (explicit overrides defaults)
    enriched_params = {**defaults, **existing_params}
    
    # Replace the original llm.invocation_parameters with enriched version
    merged_attrs["llm.invocation_parameters"] = json.dumps(enriched_params)
    
    logger.debug(f"✓ Enriched params for {provider}/{model}: added {len(defaults)} defaults")
