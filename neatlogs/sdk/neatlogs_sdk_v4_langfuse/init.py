"""
Neatlogs SDK v4 initialization.

This module handles:
- Tracer provider setup
- Span processor configuration
- Dual instrumentation (OpenInference + OpenLLMetry)
- HTTP + Threading instrumentation for context propagation
"""

import os
import time
import uuid
from typing import Optional, List
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

from .core.exporter import NeatlogsExporter
from .core.span_processor import NeatlogsSpanProcessor
from .instrumentation.manager import InstrumentationManager


_initialized = False
_tracer_provider = None
_span_processor = None
_session_config = {
    "session_id": None,
    "user_id": None,
}


def init(
    api_key: Optional[str] = None,
    endpoint: str = "http://localhost:3000/api/data/v4/batch",
    workflow_name: Optional[str] = None,
    session_id: Optional[str] = None,
    auto_session: bool = False,
    user_id: Optional[str] = None,
    
    # Instrumentation control
    instrument_tags: Optional[List[str]] = None,
    instrumentations: Optional[List[str]] = None,
    enable_http_tracing: bool = True,
    
    # Performance
    sample_rate: float = 1.0,
    batch_size: int = 100,
    flush_interval: float = 5.0,
    
    # Debug
    debug: bool = False,
) -> None:
    """
    Initialize Neatlogs SDK with dual instrumentation (OpenInference + OpenLLMetry).
    
    This sets up:
    1. OpenTelemetry tracer provider
    2. Neatlogs span processor (with attribute merging)
    3. Threading instrumentation (for context propagation)
    4. HTTP instrumentation (always-on for correct span hierarchy)
    5. Dual instrumentation for selected libraries
    
    Args:
        api_key: Neatlogs API key (default: NEATLOGS_API_KEY env var)
        endpoint: Neatlogs backend endpoint
        workflow_name: Optional workflow name for all traces
        
        session_id: Explicit session ID for chat/conversational workflows.
            Use to group multiple traces (conversation turns) into a single session.
            Reusing the same session_id across runs links them as one conversation.
        auto_session: Auto-generate unique session ID (for testing/development).
            Convenient when testing chat workflows without managing session IDs.
            Generates: session_{timestamp}_{random}
        user_id: Optional user ID for tracking
        
        instrument_tags: Semantic tags for instrumentation ["llm", "agent", "retrieval", etc.]
        instrumentations: Specific libraries to instrument ["openai", "langchain", etc.]
        enable_http_tracing: Always instrument HTTP for context propagation (recommended)
        
        sample_rate: Trace sampling rate (0.0-1.0), default 1.0 (all spans)
        batch_size: Batch export size (default 100)
        flush_interval: Seconds between batch flushes (default 5.0)
        debug: Enable debug logging
    
    Example:
        ```python
        import neatlogs
        
        # Minimal setup
        neatlogs.init(api_key="your-api-key")
        
        # Production chatbot with explicit session
        neatlogs.init(
            api_key="your-api-key",
            session_id=f"user_{user_id}_thread_{thread_id}",
            user_id=user_id,
        )
        
        # Testing chatbot with auto session
        neatlogs.init(
            api_key="your-api-key",
            auto_session=True,  # Auto-generates session ID
        )
        
        # With tags
        neatlogs.init(
            api_key="your-api-key",
            instrument_tags=["llm", "agent"],
        )
        
        # Explicit libraries
        neatlogs.init(
            api_key="your-api-key",
            instrumentations=["openai", "langchain", "chromadb"],
        )
        ```
    
    Note:
        session_id is ONLY needed for chat/conversational workflows where you want
        to group multiple traces (turns) into a session. For traditional workflows
        (data pipelines, batch jobs), omit session_id for single-trace hierarchy.
    
    Raises:
        ValueError: If API key is not provided
    """
    global _initialized
    
    if _initialized:
        if debug:
            print("⚠️  Neatlogs already initialized")
        return
    
    # Resolve API key
    api_key = api_key or os.getenv("NEATLOGS_API_KEY")
    if not api_key:
        raise ValueError(
            "api_key required. Either pass it to init() or set NEATLOGS_API_KEY environment variable."
        )
    
    # Configure logging if debug is enabled
    if debug:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(name)s - %(levelname)s - %(message)s'
        )
    
    # Determine final session ID
    final_session_id = None
    if session_id:
        # User provided explicit session
        final_session_id = session_id
    elif auto_session:
        # Auto-generate for convenience (testing/dev)
        timestamp = int(time.time())
        random_suffix = uuid.uuid4().hex[:8]
        final_session_id = f"session_{timestamp}_{random_suffix}"
        if debug:
            print(f"🔧 Auto-generated session_id: {final_session_id}")
    
    # Store session_id and user_id in global state for trace() access
    global _session_config
    _session_config["session_id"] = final_session_id
    _session_config["user_id"] = user_id
    
    # Setup resource with metadata
    # session.id and user.id are set as Resource attributes so they apply to ALL spans
    # (including orphan spans like background HTTP calls)
    resource_attrs = {
        SERVICE_NAME: workflow_name or "neatlogs-app",
        "neatlogs.workflow_name": workflow_name or "",
    }
    
    # Add session.id and user.id if present (applies globally to all spans)
    if final_session_id:
        resource_attrs["session.id"] = final_session_id
    if user_id:
        resource_attrs["user.id"] = user_id
    
    resource = Resource.create(resource_attrs)
    
    # Get or create tracer provider
    global _tracer_provider
    existing_provider = trace.get_tracer_provider()
    
    # If there's already a tracer provider (e.g., from CrewAI), use it
    # Otherwise, create a new one
    if existing_provider and hasattr(existing_provider, 'add_span_processor'):
        provider = existing_provider
        if debug:
            print("ℹ️  Using existing tracer provider (e.g., from CrewAI)")
    else:
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        if debug:
            print("✅ Created new tracer provider")
    
    _tracer_provider = provider
    
    # Create exporter
    exporter = NeatlogsExporter(
        api_key=api_key,
        endpoint=endpoint,
        batch_size=batch_size,
        flush_interval=flush_interval,
    )
    
    # Add Neatlogs span processor (with attribute merging)
    # Metrics are embedded as span attributes (neatlogs.metrics.*)
    global _span_processor
    _span_processor = NeatlogsSpanProcessor(
        exporter=exporter,
        sample_rate=sample_rate,
        debug=debug,
    )
    provider.add_span_processor(_span_processor)
    
    if debug:
        print("✅ Neatlogs tracer provider initialized")
    
    # Create instrumentation manager (exclude Neatlogs endpoint from HTTP tracing)
    manager = InstrumentationManager(
        provider=provider,
        debug=debug,
        excluded_urls=endpoint,  # Prevent infinite loop from export HTTP calls
    )
    
    # CRITICAL: Instrument threading FIRST (for context propagation)
    manager.instrument_threading()
    
    # Always instrument HTTP (critical for context propagation)
    if enable_http_tracing:
        manager.instrument_http()
    
    # Instrument based on tags and explicit libraries
    if instrument_tags or instrumentations:
        manager.instrument(
            tags=instrument_tags,
            libraries=instrumentations,
        )
        
        if debug:
            print(f"✅ Instrumented libraries: {manager.instrumented}")
    
    _initialized = True
    
    if debug:
        print("=" * 60)
        print("✅ Neatlogs SDK initialized successfully")
        print(f"   Endpoint: {endpoint}")
        print(f"   Workflow: {workflow_name or '(none)'}")
        print(f"   Session: {final_session_id or '(none)'}")
        print(f"   User: {user_id or '(none)'}")
        print(f"   Instrumentations: {manager.instrumented or '(none)'}")
        print(f"   Tags: {instrument_tags or []}")
        print(f"   Sample Rate: {sample_rate}")
        print("=" * 60)


def flush(timeout_millis: int = 30000) -> bool:
    """
    Flush all pending spans to the backend.
    
    This forces immediate export of all buffered spans.
    Metrics are embedded as span attributes.
    Useful to call before program exit to ensure all data is sent.
    
    Args:
        timeout_millis: Maximum time to wait for flush (milliseconds)
        
    Returns:
        True if flush succeeded, False otherwise
    
    Example:
        ```python
        import neatlogs
        
        neatlogs.init(api_key="...")
        
        # ... your code ...
        
        # Flush before exit
        neatlogs.flush()
        ```
    """
    global _tracer_provider
    
    success = True
    
    # Flush traces
    if _tracer_provider:
        try:
            success = _tracer_provider.force_flush(timeout_millis=timeout_millis)
        except Exception as e:
            print(f"⚠️  Error flushing spans: {e}")
            success = False
    
    return success


def get_session_config():
    """
    Get the current session configuration (session_id, user_id).
    
    Internal function used by trace() context manager.
    
    Returns:
        dict: Session configuration with 'session_id' and 'user_id' keys
    """
    return _session_config.copy()


def shutdown(timeout_millis: int = 30000) -> bool:
    """
    Shutdown the Neatlogs SDK and flush all pending spans.
    
    This should be called before program exit to ensure:
    1. All pending spans are exported (with embedded metrics)
    2. Background threads are stopped
    3. Resources are cleaned up
    
    Args:
        timeout_millis: Maximum time to wait for shutdown (milliseconds)
        
    Returns:
        True if shutdown succeeded, False otherwise
    
    Example:
        ```python
        import neatlogs
        import atexit
        
        neatlogs.init(api_key="...")
        
        # Register shutdown on exit
        atexit.register(neatlogs.shutdown)
        
        # ... or call explicitly ...
        neatlogs.shutdown()
        ```
    """
    global _tracer_provider, _span_processor, _initialized
    
    success = True
    
    # Log performance stats from span processor
    if _span_processor:
        try:
            _span_processor._log_performance_stats()
        except Exception as e:
            print(f"⚠️  Error logging performance stats: {e}")
    
    # Shutdown tracer provider
    if _tracer_provider:
        try:
            success = _tracer_provider.shutdown()
        except Exception as e:
            print(f"⚠️  Error shutting down tracer provider: {e}")
            success = False
    
    # Reset globals
    _initialized = False
    _tracer_provider = None
    _span_processor = None
    _session_config["session_id"] = None
    _session_config["user_id"] = None
    
    return success
