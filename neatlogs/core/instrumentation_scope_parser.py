"""
Framework and platform detection from instrumentation scope.

Parses OpenTelemetry instrumentation_scope.name to extract:
- Provider: LLM provider (openai, anthropic, google, etc.)
- Framework: Orchestration framework (langchain, llamaindex, crewai, etc.)
- Platform: Cloud platform (bedrock, vertex_ai, azure_openai, etc.)
"""

from typing import Dict, Optional, Tuple
import re


# Instrumentation scope patterns to framework/platform mappings
SCOPE_PATTERNS = {
    # OpenInference instrumentations
    "openinference.instrumentation.openai": {"provider": "openai"},
    "openinference.instrumentation.anthropic": {"provider": "anthropic"},
    "openinference.instrumentation.google_genai": {"provider": "google"},
    "openinference.instrumentation.bedrock": {"provider": "bedrock", "platform": "bedrock"},
    "openinference.instrumentation.vertexai": {"provider": "vertex_ai", "platform": "vertex_ai"},
    "openinference.instrumentation.mistralai": {"provider": "mistral"},
    "openinference.instrumentation.cohere": {"provider": "cohere"},
    "openinference.instrumentation.groq": {"provider": "groq"},
    
    # OpenInference frameworks
    "openinference.instrumentation.langchain": {"framework": "langchain"},
    "openinference.instrumentation.llama_index": {"framework": "llamaindex"},
    "openinference.instrumentation.llamaindex": {"framework": "llamaindex"},
    "openinference.instrumentation.crewai": {"framework": "crewai"},
    "openinference.instrumentation.haystack": {"framework": "haystack"},
    "openinference.instrumentation.dspy": {"framework": "dspy"},
    
    # OpenLLMetry (Traceloop) instrumentations
    "opentelemetry.instrumentation.openai": {"provider": "openai"},
    "opentelemetry.instrumentation.anthropic": {"provider": "anthropic"},
    "opentelemetry.instrumentation.google_generativeai": {"provider": "google"},
    "opentelemetry.instrumentation.bedrock": {"provider": "bedrock", "platform": "bedrock"},
    "opentelemetry.instrumentation.vertexai": {"provider": "vertex_ai", "platform": "vertex_ai"},
    "opentelemetry.instrumentation.cohere": {"provider": "cohere"},
    "opentelemetry.instrumentation.mistralai": {"provider": "mistral"},
    
    # OpenLLMetry frameworks
    "opentelemetry.instrumentation.langchain": {"framework": "langchain"},
    "opentelemetry.instrumentation.llamaindex": {"framework": "llamaindex"},
    "opentelemetry.instrumentation.crewai": {"framework": "crewai"},
    "opentelemetry.instrumentation.haystack": {"framework": "haystack"},
    
    # Native framework telemetry
    "haystack.telemetry": {"framework": "haystack"},
    "crewai": {"framework": "crewai"},
    "langchain": {"framework": "langchain"},
    "llama_index": {"framework": "llamaindex"},
}

# Provider-specific API variations (for more precise platform detection)
PROVIDER_VARIANTS = {
    "openai": {
        "azure": {"platform": "azure_openai"},
        "azure_openai": {"platform": "azure_openai"},
    },
    "anthropic": {
        "bedrock": {"platform": "bedrock"},
        "vertex": {"platform": "vertex_ai"},
    },
    "google": {
        "vertex": {"platform": "vertex_ai"},
        "vertexai": {"platform": "vertex_ai"},
    }
}


def parse_instrumentation_scope(scope_name: Optional[str]) -> Dict[str, str]:
    """
    Parse instrumentation scope name to extract provider/framework/platform.
    
    Args:
        scope_name: The instrumentation_scope.name from the span
        
    Returns:
        Dictionary with detected: provider, framework, platform (if found)
        
    Examples:
        "openinference.instrumentation.openai" → {"provider": "openai"}
        "openinference.instrumentation.langchain" → {"framework": "langchain"}
        "opentelemetry.instrumentation.bedrock" → {"provider": "bedrock", "platform": "bedrock"}
        "haystack.telemetry" → {"framework": "haystack"}
    """
    if not scope_name:
        return {}
    
    scope_lower = scope_name.lower()
    
    # Direct exact match
    if scope_lower in SCOPE_PATTERNS:
        return SCOPE_PATTERNS[scope_lower].copy()
    
    # Prefix match (handles versioned scopes like "openinference.instrumentation.openai.v1")
    for pattern, info in SCOPE_PATTERNS.items():
        if scope_lower.startswith(pattern):
            return info.copy()
    
    # Fuzzy extraction as fallback
    result = {}
    
    # Check for framework indicators
    if "langchain" in scope_lower:
        result["framework"] = "langchain"
    elif "llama" in scope_lower or "llamaindex" in scope_lower:
        result["framework"] = "llamaindex"
    elif "crewai" in scope_lower or "crew" in scope_lower:
        result["framework"] = "crewai"
    elif "haystack" in scope_lower:
        result["framework"] = "haystack"
    elif "dspy" in scope_lower:
        result["framework"] = "dspy"
    
    # Check for provider indicators
    if "openai" in scope_lower:
        result["provider"] = "openai"
        if "azure" in scope_lower:
            result["platform"] = "azure_openai"
    elif "anthropic" in scope_lower or "claude" in scope_lower:
        result["provider"] = "anthropic"
    elif "google" in scope_lower or "gemini" in scope_lower or "genai" in scope_lower:
        result["provider"] = "google"
    elif "bedrock" in scope_lower:
        result["provider"] = "bedrock"
        result["platform"] = "bedrock"
    elif "vertex" in scope_lower:
        result["platform"] = "vertex_ai"
        # Vertex can host multiple providers, default to google
        if "provider" not in result:
            result["provider"] = "vertex_ai"
    elif "mistral" in scope_lower:
        result["provider"] = "mistral"
    elif "cohere" in scope_lower:
        result["provider"] = "cohere"
    elif "groq" in scope_lower:
        result["provider"] = "groq"
    
    return result


def enrich_with_scope_detection(
    attrs: Dict[str, any],
    scope_name: Optional[str],
    parent_scope_name: Optional[str] = None
) -> None:
    """
    Enrich attributes with framework/platform/provider detected from instrumentation scope.
    
    Modifies attrs in-place by adding:
    - neatlogs.instrumentation.name: Original scope name
    - neatlogs.instrumentation.version: Scope version (if available)
    - neatlogs.provider: LLM provider (e.g., "openai", "anthropic")
    - neatlogs.framework: Orchestration framework (e.g., "langchain", "llamaindex")
    - neatlogs.platform: Cloud platform (e.g., "bedrock", "vertex_ai", "azure_openai")
    
    Logic:
    1. Parse current span's scope → gives provider/platform
    2. Parse parent span's scope (if provided) → gives orchestrating framework
    3. Only set attributes if not already present (explicit attrs take precedence)
    
    Args:
        attrs: Span attributes dictionary to enrich
        scope_name: Current span's instrumentation_scope.name
        parent_scope_name: Parent span's instrumentation_scope.name (optional)
    """
    # Store original instrumentation scope info
    if scope_name:
        attrs.setdefault("neatlogs.instrumentation.name", scope_name)
    
    # Parse current span's scope
    current_info = parse_instrumentation_scope(scope_name)
    
    # Set provider (from current span's scope)
    if "provider" in current_info and "neatlogs.provider" not in attrs:
        attrs["neatlogs.provider"] = current_info["provider"]
    
    # Set platform (from current span's scope)
    if "platform" in current_info and "neatlogs.platform" not in attrs:
        attrs["neatlogs.platform"] = current_info["platform"]
    
    # Set framework - prioritize parent scope, fallback to current scope
    if parent_scope_name:
        parent_info = parse_instrumentation_scope(parent_scope_name)
        if "framework" in parent_info and "neatlogs.framework" not in attrs:
            attrs["neatlogs.framework"] = parent_info["framework"]
    
    # If no framework from parent, check current scope
    if "neatlogs.framework" not in attrs and "framework" in current_info:
        attrs["neatlogs.framework"] = current_info["framework"]
    
    # Cross-reference with gen_ai.system if available
    # gen_ai.system provides the actual LLM provider used
    gen_ai_system = attrs.get("gen_ai.system", "").lower()
    if gen_ai_system and "neatlogs.provider" not in attrs:
        # Map common gen_ai.system values to providers
        provider_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google",
            "vertex_ai": "vertex_ai",
            "bedrock": "bedrock",
            "azure_openai": "openai",  # Azure OpenAI uses OpenAI provider
            "cohere": "cohere",
            "mistral": "mistral",
            "groq": "groq",
        }
        if gen_ai_system in provider_map:
            attrs["neatlogs.provider"] = provider_map[gen_ai_system]
    
    # Detect platform from llm.provider attribute (e.g., llm.provider="azure" from OpenInference)
    llm_provider = str(attrs.get("llm.provider", "")).lower()
    if llm_provider and "neatlogs.platform" not in attrs:
        if llm_provider in ("azure", "azure_openai"):
            attrs["neatlogs.platform"] = "azure_openai"

    # Detect platform from model name patterns (e.g., "anthropic.claude-3-5-sonnet-v1:0" indicates Bedrock)
    llm_model = attrs.get("llm.model_name", "")
    if llm_model and "neatlogs.platform" not in attrs:
        if llm_model.startswith("anthropic.") or llm_model.startswith("meta.") or llm_model.startswith("amazon."):
            attrs["neatlogs.platform"] = "bedrock"
        elif "azure" in llm_model.lower():
            attrs["neatlogs.platform"] = "azure_openai"


def get_effective_provider_for_pricing(attrs: Dict[str, any]) -> str:
    """
    Get the effective provider to use for pricing lookups.
    
    Logic:
    1. Use neatlogs.platform if it's set (bedrock/vertex_ai/azure_openai)
    2. Otherwise use neatlogs.provider
    3. Fallback to gen_ai.system or llm.system
    
    Returns:
        Provider string for pricing lookup (e.g., "openai", "bedrock", "anthropic")
    """
    # Platform takes precedence for pricing (different pricing on cloud platforms)
    platform = attrs.get("neatlogs.platform", "").lower()
    if platform:
        # Map platforms to pricing keys
        platform_pricing_map = {
            "bedrock": "bedrock",
            "vertex_ai": "vertex_ai",
            "azure_openai": "azure_openai",
        }
        if platform in platform_pricing_map:
            return platform_pricing_map[platform]
    
    # Use detected provider
    provider = attrs.get("neatlogs.provider", "").lower()
    if provider:
        return provider
    
    # Fallback to gen_ai.system or llm.system
    return attrs.get("gen_ai.system", attrs.get("llm.system", "")).lower()


def get_effective_provider_for_defaults(attrs: Dict[str, any]) -> str:
    """
    Get the effective provider to use for defaults lookups.
    
    Similar to pricing, but platform-specific defaults are important
    (e.g., Bedrock Claude has different parameter constraints).
    
    Returns:
        Provider string for defaults lookup (e.g., "openai", "bedrock", "anthropic")
    """
    # Same logic as pricing for now
    return get_effective_provider_for_pricing(attrs)
