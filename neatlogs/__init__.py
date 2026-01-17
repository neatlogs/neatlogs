"""
Neatlogs - LLM Call Tracking Library
==========================================

A comprehensive LLM tracking system with OpenTelemetry and OpenInference support.
Automatically captures and logs all LLM API calls with detailed metrics.
"""

from .core import LLMTracker
from . import core
import logging
import atexit
import threading
from typing import List, Optional, Dict, Any, Callable
from functools import wraps


__version__ = "1.1.7"
__all__ = [
    "init",
    "get_tracker",
    "add_tags",
    "shutdown",
    "flush",
    "set_context",
    "set_user",
    "set_session",
    "get_current_trace_id",
    "get_current_span_id",
    "annotate",
    "track_feedback",
    "trace",
    "disable_tracing",
    "enable_tracing",
]

# --- Global Tracker Instance and Initialization ---
# NOTE: We use core._global_tracker (from core.py) as the single source of truth
# to avoid having two separate _global_tracker variables

_init_lock = threading.Lock()


def get_tracker() -> Optional[LLMTracker]:
    """Get the global tracker instance."""
    return core._global_tracker


def init(
    api_key: str,
    tags: Optional[List[str]] = None,
    workflow_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug: bool = False,
    # OpenTelemetry options
    enable_otel: bool = True,
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[Dict[str, str]] = None,
    otel_console_export: bool = False,
    dry_run: bool = False,
    instrumentations: Optional[List[str]] = None,
    # Privacy & Instrumentation controls
    enable_http_tracing: bool = True,
    disable_content: bool = False,
):
    """
    Initialize the Neatlogs tracking system with optional OpenTelemetry support.

    Args:
        api_key (str): API key for the session. Will be persisted and logged.
        tags (List[str], optional): List of tags to associate with the tracking session.
        workflow_name (str, optional): Name of the workflow being tracked.
        user_id (str, optional): User identifier to associate with all traces.
        session_id (str, optional): Session identifier. Auto-generated if not provided.
        debug (bool): Enable debug logging. Defaults to False.
        enable_otel (bool): Enable OpenTelemetry tracing. Defaults to True.
        otlp_endpoint (str, optional): OTLP HTTP endpoint for exporting traces.
        otlp_headers (Dict[str, str], optional): Headers for OTLP exporter
            (e.g., {"Authorization": "Bearer xxx"}).
        otel_console_export (bool): Enable console export for debugging OTel spans.
        dry_run (bool): If True, disables sending data to Neatlogs server and enables console logging.
                        Useful for local testing and debugging. Defaults to False.
        instrumentations (List[str], optional): List of frameworks to instrument.
                                                If None, all available supported frameworks are instrumented.
                                                Supported: "openai", "openai-agents", "anthropic", "google-genai", "crewai", "groq", "litellm", "llama-index", "google-adk", "agno", "bedrock", "dspy", "guardrails", "haystack", "instructor", "mcp", "mistralai", "portkey", "pydantic-ai", "smolagents", "vertexai", "autogen-agentchat".
        enable_http_tracing (bool): Enable automatic HTTP client instrumentation (requests, httpx, aiohttp). Defaults to True.
        disable_content (bool): Disable capturing LLM prompts and completions for privacy. Defaults to False.

    Returns:
        LLMTracker: The initialized tracker instance.

    Example:
        >>> import neatlogs
        >>> # Basic usage
        >>> tracker = neatlogs.init(
        ...     api_key="your_api_key",
        ...     workflow_name="customer-support-workflow",
        ...     user_id="user_123",
        ...     session_id="session_abc"
        ... )

        >>> # With privacy controls
        >>> tracker = neatlogs.init(
        ...     api_key="your_api_key",
        ...     disable_content=True,  # Don't capture prompts/completions
        ...     enable_http_tracing=False  # Don't trace HTTP calls
        ... )

        >>> # Dry run mode (local testing)
        >>> tracker = neatlogs.init(api_key="test", dry_run=True)
    """

    agent_id = None
    thread_id = None

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    with _init_lock:
        if core._global_tracker is None:
            core._global_tracker = LLMTracker(
                api_key=api_key,
                session_id=session_id,
                agent_id=agent_id,
                thread_id=thread_id,
                user_id=user_id,
                tags=tags,
                workflow_name=workflow_name,
                # OpenTelemetry options
                enable_otel=enable_otel,
                otlp_endpoint=otlp_endpoint,
                otlp_headers=otlp_headers,
                otel_console_export=otel_console_export,
                dry_run=dry_run,
                # Privacy & Instrumentation controls
                enable_http_tracing=enable_http_tracing,
                disable_content=disable_content,
            )
            from .instrumentation import manager

            manager.instrument_all(instrumentations=instrumentations)

            # Log initialization info
            logging.info("🚀 Neatlogs Tracker initialized successfully!")
            logging.info(f"   📊 Session: {core._global_tracker.session_id}")
            if user_id:
                logging.info(f"   👤 User: {user_id}")
            logging.info(f"   🤖 Agent: {core._global_tracker.agent_id}")
            logging.info(f"   🧵 Thread: {core._global_tracker.thread_id}")
            if workflow_name:
                logging.info(f"   🌊 Workflow: {workflow_name}")
            if tags:
                logging.info(f"   🏷️  Tags: {tags}")
            if enable_otel or dry_run:
                logging.info("   📡 OpenTelemetry: Enabled")
                if otlp_endpoint:
                    logging.info(f"   🔗 OTLP Endpoint: {otlp_endpoint}")
            if enable_http_tracing:
                logging.info("   🌐 HTTP Tracing: Enabled")
            if disable_content:
                logging.info("   🔒 Content Capture: Disabled (Privacy Mode)")
            if dry_run:
                logging.info("   🧪 Dry Run: Enabled (No data sent to server)")

    return core._global_tracker


def add_tags(tags: List[str]):
    """
    Add tags to the current Neatlogs tracker.


    Args:
        tags (list): List of tags to add

    Example:
        >>> neatlogs.add_tags(["production", "customer-support", "v2.1"])
    """
    tracker = get_tracker()
    if not tracker:
        raise RuntimeError("Tracker not initialized. Call neatlogs.init() first.")

    tracker.add_tags(tags)


# --- Context Management Functions ---


def set_context(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Set contextual attributes that will be attached to all subsequent spans.
    
    This uses OpenTelemetry Baggage to propagate context across async boundaries
    and distributed traces.
    
    Args:
        user_id: User identifier to associate with traces
        session_id: Session identifier to associate with traces
        metadata: Additional custom metadata as key-value pairs
    
    Example:
        >>> neatlogs.set_context(
        ...     user_id="user_123",
        ...     session_id="session_abc",
        ...     metadata={"plan": "enterprise", "region": "us-west"}
        ... )
    """
    tracker = get_tracker()
    if not tracker:
        raise RuntimeError("Tracker not initialized. Call neatlogs.init() first.")
    
    tracker.set_context(user_id=user_id, session_id=session_id, metadata=metadata)


def set_user(user_id: str):
    """
    Set the user identifier for all subsequent traces.
    
    Args:
        user_id: User identifier
    
    Example:
        >>> neatlogs.set_user("user_123")
    """
    set_context(user_id=user_id)


def set_session(session_id: str):
    """
    Set the session identifier for all subsequent traces.
    
    Args:
        session_id: Session identifier
    
    Example:
        >>> neatlogs.set_session("session_abc")
    """
    set_context(session_id=session_id)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID.
    
    Returns:
        str: Current trace ID in hex format, or None if no active span
    
    Example:
        >>> trace_id = neatlogs.get_current_trace_id()
        >>> print(f"Current trace: {trace_id}")
    """
    from opentelemetry import trace
    
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, '032x')
    return None


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID.
    
    Returns:
        str: Current span ID in hex format, or None if no active span
    
    Example:
        >>> span_id = neatlogs.get_current_span_id()
        >>> print(f"Current span: {span_id}")
    """
    from opentelemetry import trace
    
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, '016x')
    return None


def annotate(attributes: Dict[str, Any]):
    """
    Add custom attributes to the current span.
    
    Args:
        attributes: Dictionary of attributes to add to the current span
    
    Example:
        >>> with neatlogs.trace("process_data"):
        ...     data = load_data()
        ...     neatlogs.annotate({
        ...         "record_count": len(data),
        ...         "data_source": "postgres",
        ...         "quality_score": 0.95
        ...     })
    """
    from opentelemetry import trace
    
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    else:
        logging.warning("annotate() called but no active span found")


def track_feedback(
    trace_id: Optional[str] = None,
    rating: Optional[str] = None,
    score: Optional[float] = None,
    comment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Track user feedback for a trace.
    
    Args:
        trace_id: Trace ID to attach feedback to (uses current trace if not provided)
        rating: Rating (e.g., "positive", "negative", "neutral")
        score: Numeric score (e.g., 0.0 to 1.0, or 1-5 stars)
        comment: Free-text comment
        metadata: Additional metadata
    
    Example:
        >>> # Track feedback for current trace
        >>> neatlogs.track_feedback(
        ...     rating="positive",
        ...     score=5,
        ...     comment="Great response!"
        ... )
        
        >>> # Track feedback for specific trace
        >>> trace_id = neatlogs.get_current_trace_id()
        >>> neatlogs.track_feedback(
        ...     trace_id=trace_id,
        ...     rating="negative",
        ...     comment="Incorrect answer"
        ... )
    """
    tracker = get_tracker()
    if not tracker:
        raise RuntimeError("Tracker not initialized. Call neatlogs.init() first.")
    
    # Use current trace if not provided
    if trace_id is None:
        trace_id = get_current_trace_id()
        if trace_id is None:
            logging.warning("No active trace found for feedback")
            return
    
    tracker.track_feedback(
        trace_id=trace_id,
        rating=rating,
        score=score,
        comment=comment,
        metadata=metadata,
    )


def trace(
    name: Optional[str] = None,
    enabled: bool = True,
    span_kind: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    sample_rate: float = 1.0,
    **kwargs
):
    """
    Decorator to add custom tracing to your functions.
    
    **When to use:**
    - Trace your own business logic functions
    - Add custom spans that OpenInference doesn't auto-capture
    - Control exactly what data gets captured
    
    **When NOT to use:**
    - LLM calls (auto-captured via OpenInference)
    - Framework operations (LangChain, CrewAI - auto-captured)
    - Vector DB calls (auto-captured)
    - HTTP calls (auto-captured if enable_http_tracing=True)
    
    Args:
        name: Span name (defaults to function name)
        enabled: Enable/disable tracing for this function (default: True)
        span_kind: OpenInference span kind - "CHAIN", "TOOL", "AGENT", "RETRIEVER", "EMBEDDING"
        capture_input: Capture function arguments (default: True)
        capture_output: Capture return value (default: True)
        sample_rate: Sample only X% of calls, 0.0-1.0 (default: 1.0 = 100%)
        **kwargs: Additional custom attributes
    
    Examples:
        >>> # Basic usage - trace your custom function
        >>> @neatlogs.trace()
        >>> def process_business_logic(data):
        ...     return validated_data
        
        >>> # Categorize as TOOL span
        >>> @neatlogs.trace(span_kind="TOOL")
        >>> def call_external_api():
        ...     return api_response
        
        >>> # Disable tracing for specific function
        >>> @neatlogs.trace(enabled=False)
        >>> def internal_helper():
        ...     # Not traced - useful for high-volume internal functions
        ...     pass
        
        >>> # Privacy: don't capture sensitive inputs/outputs
        >>> @neatlogs.trace(capture_input=False, capture_output=False)
        >>> def process_ssn(ssn: str):
        ...     # Function is traced but inputs/outputs not captured
        ...     return masked
        
        >>> # Sample 10% of calls (for high-volume functions)
        >>> @neatlogs.trace(sample_rate=0.1)
        >>> def frequent_operation():
        ...     return result
        
        >>> # Add custom attributes
        >>> @neatlogs.trace(operation="validation", priority="high")
        >>> def validate_data(data):
        ...     return validated
        
        >>> # As context manager
        >>> with neatlogs.trace("batch_processing"):
        ...     # Your code here
        ...     pass
    """
    from opentelemetry import trace as otel_trace
    import random
    
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **func_kwargs):
            # Check if tracing is globally disabled
            if not is_tracing_enabled():
                return func(*args, **func_kwargs)
            
            # Check if tracing is disabled for this specific function
            if not enabled:
                return func(*args, **func_kwargs)
            
            # Apply sampling (skip tracing if random roll fails)
            if sample_rate < 1.0 and random.random() > sample_rate:
                return func(*args, **func_kwargs)
            
            tracer = otel_trace.get_tracer("neatlogs")
            with tracer.start_as_current_span(span_name) as span:
                # Set OpenInference span kind if provided
                if span_kind:
                    span.set_attribute("openinference.span.kind", span_kind)
                
                # Set custom attributes
                for key, value in kwargs.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception:
                        pass  # Skip if value not serializable
                
                # Capture input arguments (if enabled)
                if capture_input:
                    try:
                        if args:
                            # Limit to first 500 chars to avoid huge spans
                            span.set_attribute("input.args", str(args)[:500])
                        if func_kwargs:
                            span.set_attribute("input.kwargs", str(func_kwargs)[:500])
                    except Exception:
                        pass  # Silently skip if serialization fails
                
                # Execute function
                try:
                    result = func(*args, **func_kwargs)
                    
                    # Capture output (if enabled)
                    if capture_output and result is not None:
                        try:
                            span.set_attribute("output.value", str(result)[:500])
                        except Exception:
                            pass
                    
                    return result
                    
                except Exception as e:
                    # Always capture errors
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    
    # Support multiple call patterns:
    # @trace, @trace(), @trace(name="x"), @trace(enabled=False), etc.
    if name is not None and not callable(name):
        # @trace(name="something") or @trace(enabled=False)
        return decorator
    elif name is None:
        # @trace() or @trace
        return decorator
    else:
        # @trace (without parentheses) - name is actually the function
        func = name
        name = None
        return decorator(func)


def flush():
    """
    Manually flush all pending spans to the backend.
    
    Useful in serverless functions or scripts where you want to ensure
    all data is sent before the process exits.
    
    Example:
        >>> neatlogs.init(api_key="...")
        >>> # ... your code ...
        >>> neatlogs.flush()  # Ensure all data is sent
    """
    tracker = get_tracker()
    if not tracker:
        raise RuntimeError("Tracker not initialized. Call neatlogs.init() first.")
    
    tracker.flush()


def shutdown():
    """
    Shutdown the Neatlogs tracker and clean up resources.
    
    This is automatically called on program exit, but you can call it manually
    if needed (e.g., in long-running processes when you're done tracking).
    
    Example:
        >>> neatlogs.init(api_key="...")
        >>> # ... your code ...
        >>> neatlogs.shutdown()
    """
    tracker = get_tracker()
    if tracker:
        tracker.shutdown()


# Global tracing state
_tracing_enabled = True
_tracing_lock = threading.Lock()


def disable_tracing():
    """
    Temporarily disable all tracing (both auto-instrumentation and decorators).
    
    Useful for:
    - High-volume background tasks where you don't need traces
    - Testing/debugging specific code paths
    - Reducing overhead in performance-critical sections
    
    Example:
        >>> neatlogs.init(api_key="...")
        >>> 
        >>> # Normal tracing
        >>> result1 = llm.invoke("query 1")  # Traced
        >>> 
        >>> # Disable tracing
        >>> neatlogs.disable_tracing()
        >>> result2 = llm.invoke("query 2")  # NOT traced
        >>> 
        >>> # Re-enable tracing
        >>> neatlogs.enable_tracing()
        >>> result3 = llm.invoke("query 3")  # Traced again
    """
    global _tracing_enabled
    with _tracing_lock:
        _tracing_enabled = False
    logging.info("🔕 Neatlogs tracing disabled")


def enable_tracing():
    """
    Re-enable tracing after it was disabled.
    
    Example:
        >>> neatlogs.disable_tracing()
        >>> # ... code not traced ...
        >>> neatlogs.enable_tracing()
        >>> # ... code traced again ...
    """
    global _tracing_enabled
    with _tracing_lock:
        _tracing_enabled = True
    logging.info("🔔 Neatlogs tracing enabled")


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _tracing_enabled


# --- Automatic Instrumentation Setup ---
# This is handled by instrument_all() called in init()


def _shutdown_neatlogs():
    """Shutdown the Neatlogs tracker and clean up resources on exit."""
    logging.debug("Neatlogs: atexit handler '_shutdown_neatlogs' called.")
    tracker = get_tracker()
    if tracker:
        tracker.shutdown()
    logging.debug("Neatlogs: atexit handler '_shutdown_neatlogs' finished.")


# Ensure that all data is sent and resources are cleaned up on exit.
atexit.register(_shutdown_neatlogs)


# Configure a default handler for the library's logger.
# This prevents "No handler found" warnings if the user of the library
# does not configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
