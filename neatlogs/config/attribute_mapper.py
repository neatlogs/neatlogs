"""
Neatlogs Semantic Conventions - Attribute Mapper

This module provides utilities to map vendor-specific attributes
(OpenInference, OpenLLMetry, GenAI) to unified neatlogs.* namespace.

Usage:
    from neatlogs.semantic_conventions import AttributeMapper
    
    mapper = AttributeMapper()
    mapped_attrs = mapper.map_attributes(original_attrs, span_kind="llm")
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class AttributeMapper:
    """
    Maps vendor-specific semantic convention attributes to Neatlogs namespace.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the attribute mapper.

        Args:
            config_path: Path to attribute-mapping.json. If None, uses default location.
        """
        if config_path is None:
            # Default to sdk/attribute-mapping.json relative to this file
            config_path = str(
                Path(__file__).parent / "attribute-mapping.json"
            )

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.mappings = self.config.get("mappings", {})
        self.keep_as_is = set(self.config.get("keep_as_is", {}).get("attributes", []))
        self.ignore_patterns = self.config.get("ignore", {}).get("patterns", [])

    def should_ignore(self, attr_name: str) -> bool:
        """Check if an attribute should be ignored."""
        for pattern in self.ignore_patterns:
            if re.match(pattern.replace("*", ".*"), attr_name):
                return True
        return False

    def should_keep_as_is(self, attr_name: str) -> bool:
        """Check if an attribute should be kept unchanged (OTEL standard)."""
        return attr_name in self.keep_as_is

    def map_span_kind(self, attributes: Dict[str, Any]) -> str:
        """
        Extract and normalize span kind from multiple possible sources.
        If no explicit span kind is found, infer from attribute patterns.

        Args:
            attributes: Original span attributes

        Returns:
            Normalized span kind (e.g., "llm", "tool", "agent")
        """
        span_kind_config = self.mappings.get("span_kind", {})
        sources = span_kind_config.get("sources", [])
        values_map = span_kind_config.get("values", {})
        priority = span_kind_config.get("priority", "openinference")

        # Try to get span kind from sources in priority order
        span_kind_value = None

        if priority == "openinference":
            # Prefer OpenInference first
            if "openinference.span.kind" in attributes:
                span_kind_value = attributes["openinference.span.kind"]
            elif "traceloop.span.kind" in attributes:
                span_kind_value = attributes["traceloop.span.kind"]
        else:
            # Try all sources
            for source in sources:
                if source in attributes:
                    span_kind_value = attributes[source]
                    break

        if span_kind_value and span_kind_value in values_map:
            return values_map[span_kind_value]

        # If no explicit span kind found, infer from attribute patterns
        # Check if this looks like an LLM span (has llm.* or gen_ai.* attributes)
        is_llm_span = any([
            "llm.model_name" in attributes,
            "gen_ai.request.model" in attributes,
            "llm.token_count.prompt" in attributes,
            "llm.token_count.completion" in attributes,
            "gen_ai.usage.prompt_tokens" in attributes,
            "gen_ai.usage.completion_tokens" in attributes,
        ])
        
        if is_llm_span:
            return "llm"

        return "unknown"

    def map_simple_attribute(
        self, mapping_config: Dict[str, Any], attributes: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Map a simple attribute from multiple sources to target.

        Args:
            mapping_config: Configuration for this attribute mapping
            attributes: Original attributes

        Returns:
            Value from first matching source, or None
        """
        sources = mapping_config.get("sources", [])

        for source in sources:
            if source in attributes:
                return attributes[source]

        return None

    def map_indexed_attributes(
        self,
        mapping_config: Dict[str, Any],
        attributes: Dict[str, Any],
        target_base: str,
    ) -> Dict[str, Any]:
        """
        Map indexed attributes like messages (llm.input_messages.0.role).

        Args:
            mapping_config: Configuration for this attribute mapping
            attributes: Original attributes
            target_base: Base target attribute name

        Returns:
            Dictionary of mapped indexed attributes
        """
        mapped = {}
        sources = mapping_config.get("sources", [])

        # Find all indexed attributes matching the pattern
        for attr_name, attr_value in attributes.items():
            for source_pattern in sources:
                # Convert {i} to regex pattern
                regex_pattern = source_pattern.replace("{i}", r"(\d+)")
                regex_pattern = regex_pattern.replace(".", r"\.")
                match = re.match(regex_pattern, attr_name)

                if match:
                    index = match.group(1)
                    # Build target attribute name
                    target = target_base.replace("{i}", index)
                    mapped[target] = attr_value

        return mapped

    def map_nested_config(
        self,
        config: Dict[str, Any],
        attributes: Dict[str, Any],
        span_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recursively map nested configuration.

        Args:
            config: Configuration dictionary
            attributes: Original attributes
            span_kind: Current span kind for dynamic substitution

        Returns:
            Mapped attributes
        """
        mapped = {}

        for key, value in config.items():
            if key in ["description", "sources", "target", "indexed", "priority", "values"]:
                # Skip metadata keys
                continue

            if isinstance(value, dict):
                if "sources" in value:
                    # This is a leaf mapping node
                    if value.get("indexed", False):
                        # Handle indexed attributes
                        target = value.get("target", "")
                        indexed_mapped = self.map_indexed_attributes(
                            value, attributes, target
                        )
                        mapped.update(indexed_mapped)

                        # Also handle target_content if present
                        if "target_content" in value:
                            content_target = value["target_content"]
                            content_mapped = self.map_indexed_attributes(
                                value, attributes, content_target
                            )
                            mapped.update(content_mapped)
                    else:
                        # Simple attribute mapping
                        target = value.get("target", "")
                        if span_kind and "{span_kind}" in target:
                            target = target.replace("{span_kind}", span_kind)

                        result = self.map_simple_attribute(value, attributes)
                        if result is not None:
                            mapped[target] = result
                else:
                    # Recursive nested mapping
                    nested_mapped = self.map_nested_config(value, attributes, span_kind)
                    mapped.update(nested_mapped)

        return mapped

    def map_attributes(
        self, attributes: Dict[str, Any], span_kind: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Map all attributes from vendor-specific to neatlogs namespace.

        Args:
            attributes: Original span attributes
            span_kind: Optional span kind override. If None, will be extracted from attributes.

        Returns:
            Mapped attributes with neatlogs.* namespace
        """
        mapped = {}

        # 1. Extract span kind
        if span_kind is None:
            span_kind = self.map_span_kind(attributes)
        mapped["neatlogs.span.kind"] = span_kind

        # 2. Map all configured attributes
        for section_name, section_config in self.mappings.items():
            if section_name == "span_kind":
                # Already handled
                continue

            if isinstance(section_config, dict) and "mappings" in section_config:
                # Has nested mappings
                nested_mapped = self.map_nested_config(
                    section_config["mappings"], attributes, span_kind
                )
                mapped.update(nested_mapped)
            elif isinstance(section_config, dict) and "sources" in section_config:
                # Direct mapping
                target = section_config.get("target", "")
                if "{span_kind}" in target:
                    target = target.replace("{span_kind}", span_kind)

                result = self.map_simple_attribute(section_config, attributes)
                if result is not None:
                    mapped[target] = result
            else:
                # Nested config
                nested_mapped = self.map_nested_config(
                    section_config, attributes, span_kind
                )
                mapped.update(nested_mapped)

        # 3. Keep OpenTelemetry standard attributes as-is
        for attr_name, attr_value in attributes.items():
            if self.should_keep_as_is(attr_name) and attr_name not in mapped:
                mapped[attr_name] = attr_value

        # 4. Preserve unmapped custom attributes (don't drop user-defined attributes)
        # Track which source attributes were mapped
        mapped_sources = set()
        for section_config in self.mappings.values():
            if isinstance(section_config, dict):
                if "sources" in section_config:
                    mapped_sources.update(section_config.get("sources", []))
                elif "mappings" in section_config:
                    self._collect_mapped_sources(section_config["mappings"], mapped_sources)
                else:
                    self._collect_mapped_sources(section_config, mapped_sources)
        
        # Add unmapped attributes (except ignored ones)
        for attr_name, attr_value in attributes.items():
            if (
                attr_name not in mapped_sources  # Not a source attribute that was mapped
                and attr_name not in mapped  # Not already in mapped dict
                and not self.should_ignore(attr_name)  # Not in ignore list
            ):
                mapped[attr_name] = attr_value

        # 5. Filter out ignored attributes from final result
        mapped = {
            k: v for k, v in mapped.items() if not self.should_ignore(k)
        }

        return mapped
    
    def _collect_mapped_sources(self, config: Dict[str, Any], collected: set) -> None:
        """
        Recursively collect all source attribute names from nested config.
        
        Args:
            config: Configuration dict (potentially nested)
            collected: Set to add source attribute names to
        """
        for value in config.values():
            if isinstance(value, dict):
                if "sources" in value:
                    collected.update(value.get("sources", []))
                else:
                    self._collect_mapped_sources(value, collected)

    def get_span_kind_value_mapping(self) -> Dict[str, str]:
        """Get the span kind value mapping."""
        return self.mappings.get("span_kind", {}).get("values", {})

    def get_target_attribute_name(
        self, source_attr: str, span_kind: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the target neatlogs attribute name for a source attribute.

        Args:
            source_attr: Source attribute name
            span_kind: Optional span kind for dynamic substitution

        Returns:
            Target attribute name or None if no mapping exists
        """
        # Simple reverse lookup - could be optimized with a reverse index
        def search_config(config: Dict[str, Any], target_span_kind: Optional[str]) -> Optional[str]:
            for key, value in config.items():
                if isinstance(value, dict):
                    if "sources" in value:
                        sources = value.get("sources", [])
                        if source_attr in sources:
                            target = value.get("target", "")
                            if target_span_kind and "{span_kind}" in target:
                                target = target.replace("{span_kind}", target_span_kind)
                            return target
                    # Recursive search
                    result = search_config(value, target_span_kind)
                    if result:
                        return result
            return None

        return search_config(self.mappings, span_kind)


# Global instance
_mapper_instance: Optional[AttributeMapper] = None


def get_mapper() -> AttributeMapper:
    """Get the global AttributeMapper instance."""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = AttributeMapper()
    return _mapper_instance


def map_attributes(
    attributes: Dict[str, Any], span_kind: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to map attributes using the global mapper.

    Args:
        attributes: Original span attributes
        span_kind: Optional span kind override

    Returns:
        Mapped attributes with neatlogs.* namespace
    """
    mapper = get_mapper()
    return mapper.map_attributes(attributes, span_kind)


if __name__ == "__main__":
    # Example usage / testing
    mapper = AttributeMapper()

    # Test data
    test_attrs = {
        "openinference.span.kind": "LLM",
        "traceloop.span.kind": "task",
        "llm.model_name": "gpt-4o-mini-2024-07-18",
        "llm.provider": "openai",
        "llm.token_count.total": 1250,
        "llm.token_count.prompt": 1000,
        "llm.token_count.completion": 250,
        "llm.cost.total": 0.003,
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "Hello!",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": "Hello!",
        "input.value": "test input",
        "traceloop.entity.input": "test input",
        "session.id": "session_123",
        "http.method": "POST",
        "http.url": "https://api.openai.com/v1/chat/completions",
        "service.name": "my-app",
        "telemetry.distro.name": "openllmetry",
    }

    print("Original attributes:")
    print(json.dumps(test_attrs, indent=2))
    print("\n" + "=" * 80 + "\n")

    mapped = mapper.map_attributes(test_attrs)

    print("Mapped attributes:")
    print(json.dumps(mapped, indent=2))
    print("\n" + "=" * 80 + "\n")

    print(f"Total attributes: {len(test_attrs)} → {len(mapped)}")
    print(f"Span kind detected: {mapped.get('neatlogs.span.kind')}")
