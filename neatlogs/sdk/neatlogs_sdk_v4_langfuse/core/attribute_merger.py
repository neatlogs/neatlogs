"""
Smart attribute merger for OpenInference + OpenLLMetry conventions.

Strategy:
1. Map duplicate attributes to canonical (OpenInference) names
2. Preserve unique attributes from both conventions
3. Calculate derived attributes (cost, totals, etc.)
"""

import json
import os
from typing import Dict, Any, Optional


class AttributeMerger:
    """
    Merges attributes from OpenInference + OpenLLMetry instrumentations.
    
    Handles:
    - Deduplication of overlapping attributes
    - Preservation of unique attributes from both conventions
    - Cost calculation when token counts are available (uses pricing.json)
    - Normalization to OpenInference canonical format
    """
    
    # Load pricing data from JSON file
    _pricing_data: Optional[Dict[str, Any]] = None
    
    @classmethod
    def _load_pricing(cls) -> Dict[str, Any]:
        """Load pricing data from pricing.json file."""
        if cls._pricing_data is None:
            pricing_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),  # Go up from core/
                "pricing.json"
            )
            try:
                with open(pricing_file, 'r') as f:
                    cls._pricing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load pricing.json: {e}")
                cls._pricing_data = {"chat": {}, "embeddings": {}}
        return cls._pricing_data
    
    # Map: OpenLLMetry/GenAI attribute → OpenInference attribute (canonical)
    ATTRIBUTE_MAPPING = {
        # Token counts (GenAI semantic conventions)
        "gen_ai.usage.prompt_tokens": "llm.token_count.prompt",
        "gen_ai.usage.completion_tokens": "llm.token_count.completion",
        "gen_ai.usage.input_tokens": "llm.token_count.prompt",
        "gen_ai.usage.output_tokens": "llm.token_count.completion",
        
        # Cache tokens (Anthropic-specific)
        "gen_ai.usage.cache_read_input_tokens": "llm.token_count.prompt.cache_read",
        "gen_ai.usage.cache_creation_input_tokens": "llm.token_count.prompt.cache_creation",
        "llm.usage.cache_read_input_tokens": "llm.token_count.prompt.cache_read",
        "llm.usage.cache_creation_input_tokens": "llm.token_count.prompt.cache_creation",
        
        # Total tokens
        "llm.usage.total_tokens": "llm.token_count.total",
        
        # Model name
        "gen_ai.request.model": "llm.model_name",
        "gen_ai.response.model": "llm.model_name",
        
        # System/provider
        "gen_ai.system": "llm.system",
        
        # Temperature, top_p, max_tokens
        "gen_ai.request.temperature": "llm.invocation_parameters",
        "gen_ai.request.top_p": "llm.invocation_parameters",
        "gen_ai.request.max_tokens": "llm.invocation_parameters",
        
        # Messages (prompt and completion)
        "gen_ai.prompt": "llm.input_messages",
        "gen_ai.completion": "llm.output_messages",
    }
    
    # Attributes ONLY in OpenLLMetry (preserve these)
    OPENLLMETRY_UNIQUE = {
        "llm.is_streaming",
        "llm.response.finish_reason",
        "llm.response.stop_reason",
        "traceloop.span.kind",
        "traceloop.entity.name",
        "traceloop.entity.path",
        "traceloop.entity.version",
        "traceloop.entity.input",
        "traceloop.entity.output",
        "traceloop.workflow.name",
        "traceloop.association.properties",
        "traceloop.prompt.managed",
        "traceloop.prompt.key",
        "traceloop.prompt.version",
        "traceloop.prompt.version_name",
        "traceloop.prompt.version_hash",
        "traceloop.prompt.template",
        "traceloop.prompt.template_variables",
        # Streaming latency metrics (OpenLit)
        "gen_ai.server.time_per_output_token",
        "completion_start_time",
    }
    
    # Attributes ONLY in OpenInference (preserve these)
    OPENINFERENCE_UNIQUE = {
        "openinference.span.kind",
        "llm.cost.total",
        "llm.cost.prompt",
        "llm.cost.completion",
        "llm.function_call",
        "llm.tools",
        "llm.prompt_template",
        "llm.prompt_template_variables",
        "llm.prompt_template.version",
        "embedding.embeddings",
        "embedding.model_name",
        "embedding.vector",
        "retrieval.documents",
        "reranker.model_name",
        "reranker.query",
        "reranker.top_k",
        # Derived latency metrics (calculated by Neatlogs)
        "llm.time_to_first_token",
        "llm.streaming_latency",
        "llm.output_tokens_per_second",
        "llm.tokens_per_second",
    }
    
    def merge(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge attributes from both conventions into canonical OpenInference format.
        
        Args:
            attributes: Raw attributes from span (may contain duplicates from both conventions)
            
        Returns:
            Deduplicated attributes with OpenInference canonical names + unique from both
        """
        merged = {}
        processed = set()
        
        # Step 1: Map GenAI/OpenLLMetry → OpenInference (canonical)
        for source_key, target_key in self.ATTRIBUTE_MAPPING.items():
            if source_key in attributes:
                value = attributes[source_key]
                
                # Special handling for invocation parameters (merge into JSON)
                if target_key == "llm.invocation_parameters":
                    if target_key not in merged:
                        merged[target_key] = {}
                    # Extract parameter name from source key
                    param_name = source_key.split(".")[-1]  # e.g., "temperature"
                    merged[target_key][param_name] = value
                else:
                    merged[target_key] = value
                
                processed.add(source_key)
                processed.add(target_key)
            elif target_key in attributes:
                # OpenInference attribute already present
                merged[target_key] = attributes[target_key]
                processed.add(target_key)
        
        # Convert invocation parameters dict to JSON string if present
        if "llm.invocation_parameters" in merged and isinstance(merged["llm.invocation_parameters"], dict):
            merged["llm.invocation_parameters"] = json.dumps(merged["llm.invocation_parameters"])
        
        # Step 2: Add unique OpenLLMetry attributes
        for key in self.OPENLLMETRY_UNIQUE:
            if key in attributes and key not in processed:
                merged[key] = attributes[key]
                processed.add(key)
        
        # Step 3: Add unique OpenInference attributes
        for key in self.OPENINFERENCE_UNIQUE:
            if key in attributes and key not in processed:
                merged[key] = attributes[key]
                processed.add(key)
        
        # Step 4: Add any remaining attributes (custom user attributes, vector DB attributes, etc.)
        for key, value in attributes.items():
            if key not in processed:
                # Keep all other attributes as-is
                merged[key] = value
        
        # Step 5: Calculate derived attributes (cost, totals)
        self._calculate_derived(merged)
        
        return merged
    
    def _calculate_derived(self, merged: Dict[str, Any]) -> None:
        """
        Calculate derived attributes if missing.
        
        Modifies merged dict in-place.
        """
        # Calculate total tokens if not present
        if "llm.token_count.total" not in merged:
            prompt_tokens = merged.get("llm.token_count.prompt", 0)
            completion_tokens = merged.get("llm.token_count.completion", 0)
            if prompt_tokens or completion_tokens:
                merged["llm.token_count.total"] = prompt_tokens + completion_tokens
        
        # Calculate cost if tokens present but cost missing
        if "llm.cost.total" not in merged:
            model = merged.get("llm.model_name", "")
            prompt_tokens = merged.get("llm.token_count.prompt", 0)
            completion_tokens = merged.get("llm.token_count.completion", 0)
            
            if model and (prompt_tokens or completion_tokens):
                cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
                if cost:
                    merged["llm.cost.prompt"] = cost["prompt"]
                    merged["llm.cost.completion"] = cost["completion"]
                    merged["llm.cost.total"] = cost["total"]
    
    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> Optional[Dict[str, float]]:
        """
        Calculate cost based on model pricing from pricing.json.
        
        This is a FALLBACK - only used if OpenInference didn't provide cost.
        Pricing is loaded from pricing.json (per 1K tokens in USD).
        """
        pricing_data = self._load_pricing()
        chat_pricing = pricing_data.get("chat", {})
        
        model_lower = model.lower()
        
        # Try exact match first
        if model in chat_pricing:
            prices = chat_pricing[model]
            prompt_cost = (prompt_tokens / 1000) * prices["promptPrice"]
            completion_cost = (completion_tokens / 1000) * prices["completionPrice"]
            return {
                "prompt": round(prompt_cost, 6),
                "completion": round(completion_cost, 6),
                "total": round(prompt_cost + completion_cost, 6),
            }
        
        # Try fuzzy match (handle versions like gpt-4-0125-preview)
        for model_key, prices in chat_pricing.items():
            if model_key.lower() in model_lower or model_lower.startswith(model_key.lower()):
                prompt_cost = (prompt_tokens / 1000) * prices["promptPrice"]
                completion_cost = (completion_tokens / 1000) * prices["completionPrice"]
                return {
                    "prompt": round(prompt_cost, 6),
                    "completion": round(completion_cost, 6),
                    "total": round(prompt_cost + completion_cost, 6),
                }
        
        # No pricing found
        return None
