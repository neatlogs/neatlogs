"""
Neatlogs span processor with attribute merging and smart export.
"""

import random
import logging
import time
from typing import Optional
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, Span
from opentelemetry.context import Context
from opentelemetry import baggage

from .attribute_merger import AttributeMerger
from .exporter import NeatlogsExporter
from ..span_kinds.mapping import OPENINFERENCE_TO_TRACELOOP, TRACELOOP_TO_OPENINFERENCE

logger = logging.getLogger(__name__)


class NeatlogsSpanProcessor(SpanProcessor):
    """
    Neatlogs span processor with:
    1. Attribute merging (OpenInference + OpenLLMetry → canonical OpenInference)
    2. Span kind normalization (OpenInference primary, Traceloop secondary)
    3. Sampling for production efficiency
    4. Batched export to Neatlogs backend
    """
    
    def __init__(
        self,
        exporter: NeatlogsExporter,
        sample_rate: float = 1.0,
        debug: bool = False,
    ):
        """
        Initialize the span processor.
        
        Args:
            exporter: Neatlogs exporter instance
            sample_rate: Fraction of spans to export (0.0-1.0). Default 1.0 (all spans).
            debug: Enable debug logging
        """
        self.exporter = exporter
        self.sample_rate = sample_rate
        self.merger = AttributeMerger()
        self.debug = debug
        
        # Performance tracking (always on)
        self.perf_stats = {
            "on_start_time": 0.0,  # Total time in on_start (seconds)
            "on_end_time": 0.0,    # Total time in on_end (seconds)
            "spans_processed": 0,   # Total spans processed
            "spans_exported": 0,    # Total spans exported
        }
        
        # Set logger level based on debug flag
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """
        Called when span starts.
        
        For LLM spans, we read prompt variables from the context (set by @observe decorator)
        and propagate them to the LLM span.
        
        This ensures prompt variables are ALWAYS on LLM spans, not wrapper spans,
        making backend queries consistent and predictable.
        
        Args:
            span: The span that's starting
            parent_context: The parent context containing prompt metadata
        """
        start_time = time.perf_counter()
        
        # Check if this will be an LLM span (based on span name or existing attributes)
        # We need to check early, but span.attributes might not be fully set yet
        # Most LLM instrumentations set openinference.span.kind early
        span_kind = span.attributes.get("openinference.span.kind") if span.attributes else None
        
        # Also check span name patterns for LLM operations
        is_llm_span = (
            span_kind == "LLM" or
            "chat" in span.name.lower() or
            "completion" in span.name.lower() or
            "generate" in span.name.lower() or
            "embedding" in span.name.lower()
        )
        
        if not is_llm_span:
            return
        
        # Read prompt metadata from context (set by @observe decorator using context.set_value)
        from opentelemetry.context import get_value, get_current
        
        ctx = parent_context if parent_context else get_current()
        
        variables_json = get_value("neatlogs.prompt_variables", context=ctx)
        template = get_value("neatlogs.prompt_template", context=ctx)
        version_val = get_value("neatlogs.prompt_version", context=ctx)
        
        # Debug logging
        logger.debug(f"[SpanProcessor.on_start] LLM span '{span.name}' starting")
        logger.debug(f"  parent_context: {parent_context is not None}")
        logger.debug(f"  variables_json from context: {variables_json}")
        logger.debug(f"  template from context: {template}")
        logger.debug(f"  version from context: {version_val}")
        
        # Set on LLM span (only if not already set by instrumentation)
        if variables_json and "llm.prompt_template_variables" not in (span.attributes or {}):
            span.set_attribute("llm.prompt_template_variables", variables_json)
            logger.debug(f"  ✓ Set llm.prompt_template_variables")
        
        if template and "llm.prompt_template" not in (span.attributes or {}):
            span.set_attribute("llm.prompt_template", template)
            logger.debug(f"  ✓ Set llm.prompt_template")
        
        if version_val and "llm.prompt_template.version" not in (span.attributes or {}):
            span.set_attribute("llm.prompt_template.version", version_val)
            logger.debug(f"  ✓ Set llm.prompt_template.version")
        
        # Track performance
        self.perf_stats["on_start_time"] += time.perf_counter() - start_time
    
    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when span ends.
        
        This is where we:
        1. Apply sampling (skip low-value spans)
        2. Merge attributes from both conventions
        3. Normalize span kinds
        4. Export to Neatlogs backend
        """
        start_time = time.perf_counter()
        self.perf_stats["spans_processed"] += 1
        
        logger.debug(f"[SpanProcessor.on_end] Span ending: {span.name} (parent: {span.parent.span_id if span.parent else None})")
        
        # Sampling - skip spans based on sample rate
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            logger.debug(f"  ⏭️  Skipped due to sampling")
            self.perf_stats["on_end_time"] += time.perf_counter() - start_time
            return
        
        # Get raw attributes from span
        raw_attrs = dict(span.attributes) if span.attributes else {}
        
        # Merge in Resource attributes (session.id, user.id, etc.)
        # Resource attributes are set globally in init() and apply to ALL spans
        if span.resource and span.resource.attributes:
            resource_attrs = dict(span.resource.attributes)
            logger.debug(f"  Resource attributes: {list(resource_attrs.keys())}")
            for key, value in resource_attrs.items():
                # Only add if not already in span attributes (span attributes take precedence)
                if key not in raw_attrs:
                    raw_attrs[key] = value
                    logger.debug(f"    ✓ Merged Resource attr: {key} = {value}")
        
        # Merge attributes (deduplicate, preserve unique, calculate derived)
        merged_attrs = self.merger.merge(raw_attrs)
        
        # Calculate derived latency and throughput metrics
        self._calculate_latency_metrics(span, merged_attrs)
        
        # Enrich invocation parameters with model defaults (if enabled)
        from ..config import enrich_invocation_parameters
        enrich_invocation_parameters(merged_attrs, enable_enrichment=True)
        
        # Get span kind (OpenInference primary for AI/LLM, OTel native for HTTP)
        # Rule: Only HTTP spans (CLIENT/SERVER) use native OTel kinds, all others need semantic kinds
        
        # Get what both conventions say
        openinference_span_kind = merged_attrs.get("openinference.span.kind")
        traceloop_span_kind = merged_attrs.get("traceloop.span.kind")
        native_otel_kind = span.kind.name  # e.g., "CLIENT", "SERVER", "INTERNAL"
        
        # Check if this is an HTTP span (has http.* attributes)
        is_http_span = "http.method" in merged_attrs or "http.url" in merged_attrs
        
        # Check if this looks like an LLM/AI span (has llm.* or gen_ai.* attributes)
        is_llm_span = any([
            "llm.model_name" in merged_attrs,
            "gen_ai.request.model" in merged_attrs,
            "llm.token_count.prompt" in merged_attrs,
            "llm.token_count.completion" in merged_attrs,
            "gen_ai.usage.prompt_tokens" in merged_attrs,
            "gen_ai.usage.completion_tokens" in merged_attrs,
        ])
        
        # Debug logging
        logger.debug(f"[SpanProcessor] Span: {span.name}")
        logger.debug(f"  OpenInference kind: {openinference_span_kind}")
        logger.debug(f"  OpenLLMetry kind: {traceloop_span_kind}")
        logger.debug(f"  Native OTel kind: {native_otel_kind}")
        logger.debug(f"  Is HTTP span: {is_http_span}")
        logger.debug(f"  Is LLM span: {is_llm_span}")
        
        # If no OpenInference kind but Traceloop kind exists, convert it
        if not openinference_span_kind:
            traceloop_kind = merged_attrs.get("traceloop.span.kind")
            if traceloop_kind:
                openinference_span_kind = TRACELOOP_TO_OPENINFERENCE.get(traceloop_kind, "CHAIN")
                merged_attrs["openinference.span.kind"] = openinference_span_kind
        
        # Add Traceloop span kind if missing (for compatibility)
        if openinference_span_kind and "traceloop.span.kind" not in merged_attrs:
            traceloop_kind = OPENINFERENCE_TO_TRACELOOP.get(openinference_span_kind, "task")
            merged_attrs["traceloop.span.kind"] = traceloop_kind
        
        # Determine the kind field for export
        # Priority:
        # 1. OpenInference semantic kind (if set by instrumentation)
        # 2. HTTP spans use native OTel kind (CLIENT, SERVER, etc.)
        # 3. LLM spans without semantic kind → infer as "LLM"
        # 4. Otherwise → UNKNOWN
        if openinference_span_kind:
            export_kind = openinference_span_kind
            logger.debug(f"  → Using OpenInference kind: {export_kind}")
        elif is_http_span:
            # Pure HTTP spans (no LLM attributes) use native OTel kind
            export_kind = str(span.kind)  # Will be "SpanKind.CLIENT", etc.
            logger.debug(f"  → Using native OTel kind for HTTP: {export_kind}")
        elif is_llm_span:
            # Has LLM attributes but no semantic kind → infer as LLM
            export_kind = "LLM"
            merged_attrs["openinference.span.kind"] = "LLM"  # Add to merged attrs for consistency
            logger.debug(f"  → Inferred as LLM (has llm.* attributes)")
        else:
            # Neither convention set a kind, default to UNKNOWN
            export_kind = "UNKNOWN"
            logger.debug(f"  → No semantic kind from instrumentations, using UNKNOWN")
        
        # Build span data for export
        span_data = {
            "trace_id": f"{span.context.trace_id:032x}",
            "span_id": f"{span.context.span_id:016x}",
            "parent_span_id": (
                f"{span.parent.span_id:016x}" if span.parent else None
            ),
            "name": span.name,
            "kind": export_kind,  # OpenInference kind OR OTel native kind
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ns": span.end_time - span.start_time if span.end_time else None,
            "attributes": merged_attrs,
            "status": {
                "code": span.status.status_code.name,
                "description": span.status.description,
            },
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes) if event.attributes else {},
                }
                for event in span.events
            ] if span.events else [],
        }
        
        logger.debug(f"  ✓ Exporting span: {span.name} [{export_kind}] (parent: {span_data['parent_span_id']})")
        
        # Export (batched, async)
        self.exporter.export(span_data)
        self.perf_stats["spans_exported"] += 1
        
        # Track performance
        self.perf_stats["on_end_time"] += time.perf_counter() - start_time
    
    def _calculate_latency_metrics(
        self, span: ReadableSpan, merged_attrs: dict
    ) -> None:
        """
        Calculate derived latency and throughput metrics.
        
        Stores as span attributes (neatlogs.metrics.*) for per-request analysis.
        
        Calculates (if applicable):
        - llm.is_streaming: Boolean flag indicating streaming response (OpenLLMetry standard)
        - neatlogs.metrics.time_to_first_token: TTFT in ms (span attribute)
        - neatlogs.metrics.streaming_latency: Streaming latency in ms (span attribute)
        - neatlogs.metrics.output_tokens_per_second: Output tokens/sec (span attribute)
        - neatlogs.metrics.tokens_per_second: Total tokens/sec (span attribute)
        
        Plus OTel histogram metrics:
        - gen_ai.server.time_to_first_token (seconds)
        - llm.chat_completions.streaming_time_to_generate (seconds)
        - gen_ai.client.operation.duration (seconds)
        - neatlogs.llm.output_tokens_per_second
        - neatlogs.llm.tokens_per_second
        
        Sources for completion_start_time:
        - Span events only (OpenInference "First Token Stream Event")
        
        Args:
            span: The span being processed
            merged_attrs: Merged attributes dict (modified in-place)
        """
        # Get timestamps (nanoseconds)
        start_time_ns = span.start_time
        end_time_ns = span.end_time
        
        if not end_time_ns or not start_time_ns:
            return
        
        # Get completion_start_time from span events
        completion_start_time_ns = self._get_completion_start_time(span, merged_attrs)
        
        # Calculate latency in seconds (for metrics and token per second calculations)
        latency_s = (end_time_ns - start_time_ns) / 1_000_000_000
        
        # Prepare metric attributes (dimensions for histograms)
        metric_attrs = self._get_metric_attributes(span, merged_attrs)
        
        # Detect and set streaming flag (OpenLLMetry standard attribute)
        is_streaming = self._is_streaming_response(merged_attrs, completion_start_time_ns)
        if is_streaming:
            merged_attrs["llm.is_streaming"] = True
        
        # Calculate streaming metrics (if we have first token timestamp)
        if completion_start_time_ns:
            # 1. Time to First Token (TTFT)
            ttft_s = (completion_start_time_ns - start_time_ns) / 1_000_000_000
            ttft_ms = ttft_s * 1000
            merged_attrs["neatlogs.metrics.time_to_first_token"] = round(ttft_ms, 3)
            
            # 2. Streaming Latency (first token to completion)
            streaming_latency_s = (end_time_ns - completion_start_time_ns) / 1_000_000_000
            streaming_latency_ms = streaming_latency_s * 1000
            merged_attrs["neatlogs.metrics.streaming_latency"] = round(streaming_latency_ms, 3)
            
            # 3. Output Tokens Per Second (only meaningful for streaming)
            output_tokens = merged_attrs.get("llm.token_count.completion", 0)
            if output_tokens and streaming_latency_s > 0:
                output_tokens_per_sec = output_tokens / streaming_latency_s
                merged_attrs["neatlogs.metrics.output_tokens_per_second"] = round(output_tokens_per_sec, 2)
        
        # 4. Total Tokens Per Second (always calculate for LLM/AI spans)
        total_tokens = merged_attrs.get("llm.token_count.total", 0)
        if total_tokens and latency_s > 0:
            tokens_per_sec = total_tokens / latency_s
            merged_attrs["neatlogs.metrics.tokens_per_second"] = round(tokens_per_sec, 2)
    
    def _get_metric_attributes(self, span: ReadableSpan, merged_attrs: dict) -> dict:
        """
        Extract attributes for histogram metrics (dimensions).
        
        Includes trace_id and span_id for correlation with spans.
        
        Args:
            span: The span being processed
            merged_attrs: Merged span attributes
            
        Returns:
            Dictionary of metric attributes
        """
        # Get trace_id and span_id from span context
        trace_id = format(span.context.trace_id, "032x") if span.context else "unknown"
        span_id = format(span.context.span_id, "016x") if span.context else "unknown"
        
        # Extract model name (check multiple possible attribute names)
        model = (
            merged_attrs.get("gen_ai.response.model") or 
            merged_attrs.get("llm.model_name") or 
            merged_attrs.get("llm.model") or 
            "unknown"
        )
        
        # Extract server address from http.url if present
        http_url = merged_attrs.get("http.url")
        if http_url:
            # Parse domain from URL (e.g., "https://api.openai.com/v1/chat/completions" -> "api.openai.com")
            try:
                from urllib.parse import urlparse
                parsed = urlparse(http_url)
                server_address = parsed.netloc or "unknown"
            except Exception:
                server_address = "unknown"
        else:
            server_address = "unknown"
        
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "gen_ai.response.model": model,
            "server.address": server_address,
        }
    
    def _get_completion_start_time(self, span: ReadableSpan, merged_attrs: dict) -> Optional[int]:
        """
        Get the timestamp when the first token was generated (completion start time).
        
        Checks span events for first token timestamp.
        
        Args:
            span: The span being processed
            merged_attrs: Merged attributes dict
            
        Returns:
            Timestamp in nanoseconds, or None if not found
        """
        # Try to extract from span events (primary method)
        if span.events:
            completion_start_time_ns = self._extract_first_token_time_from_events(span.events)
            if completion_start_time_ns:
                return completion_start_time_ns
        
        return None
    
    def _is_streaming_response(self, merged_attrs: dict, completion_start_time_ns: Optional[int]) -> bool:
        """
        Detect if this is a streaming response.
        
        Args:
            merged_attrs: Merged span attributes
            completion_start_time_ns: First token timestamp (if available)
            
        Returns:
            True if this is a streaming response
        """
        # If we have a completion_start_time, it's likely streaming
        if completion_start_time_ns:
            return True
        
        # Check for explicit streaming flags in attributes
        if merged_attrs.get("llm.is_streaming") is True:
            return True
        
        # Check for streaming-related attributes
        if "stream" in str(merged_attrs.get("gen_ai.request.stream", "")).lower():
            return True
        
        return False
    
    def _extract_first_token_time_from_events(self, events) -> Optional[int]:
        """
        Extract first token timestamp from span events.
        
        OpenInference uses span events to track when the first token arrives:
        - Event name: "First Token Stream Event"
        - Other instrumentations may use events with "token" in the name
        
        Args:
            events: List of span events
            
        Returns:
            Timestamp in nanoseconds, or None if not found
        """
        for event in events:
            event_name = event.name.lower()
            # Check for OpenInference-style event
            if "first token" in event_name:
                return event.timestamp
            # Check for generic token event (must be early in the span)
            elif "token" in event_name and "new" in event_name:
                return event.timestamp
        
        return None
    
    def _log_performance_stats(self) -> None:
        """Log performance statistics showing Neatlogs overhead."""
        stats = self.perf_stats
        
        if stats["spans_processed"] == 0:
            return
        
        total_time = stats["on_start_time"] + stats["on_end_time"]
        avg_time_per_span = (total_time / stats["spans_processed"]) * 1000
        
        print(f"\n📊 Neatlogs Overhead: {total_time*1000:.2f}ms total, {avg_time_per_span:.3f}ms/span ({stats['spans_processed']} spans)")
    
    def shutdown(self) -> None:
        """
        Shutdown the processor and flush remaining spans.
        """
        # Log performance stats
        self._log_performance_stats()
        
        self.exporter.shutdown()
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush all pending spans.
        
        Args:
            timeout_millis: Maximum time to wait (milliseconds)
            
        Returns:
            True if flush succeeded
        """
        try:
            self.exporter.flush(timeout=timeout_millis / 1000.0)
            return True
        except Exception:
            return False
