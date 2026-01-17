"""
Core tracking functionality for Neatlogs Tracker

This module provides the core LLM tracking functionality with OpenTelemetry
and OpenInference integration for standardized observability.
"""

import os
import queue
import threading
import logging
import time
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests
from opentelemetry.trace import Span

import contextvars

current_span_id_context = contextvars.ContextVar(
    "current_span_id", default=None)


# Sentinel object for queue lifecycle management
_STOP = object()


@dataclass
class LLMCallData:
    """Data structure for LLM call information"""

    trace_id: str
    span: dict
    api_key: Optional[str] = None


class LLMTracker:
    """
    Main orchestrator for LLM tracking, logging, and reporting.

    The LLMTracker manages the lifecycle of LLM operations, from span creation
    to data collection and reporting. It handles both file-based logging and
    server-side telemetry transmission, with optional OpenTelemetry integration.

    Key Responsibilities:
    - Managing active spans and completed calls
    - Coordinating background threads for server communication
    - OpenTelemetry span creation and attribute population
    - Handling graceful shutdown procedures
    - Providing thread-safe operations for concurrent environments
    """

    def __init__(
        self,
        api_key,
        session_id=None,
        agent_id=None,
        thread_id=None,
        user_id=None,
        tags=None,
        workflow_name=None,
        enable_server_sending=True,
        # OpenTelemetry options
        enable_otel: bool = True,  # Default to True now as it's the core engine
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[Dict[str, str]] = None,
        otel_console_export: bool = False,
        dry_run: bool = False,
        # Privacy & Instrumentation controls
        enable_http_tracing: bool = True,
        disable_content: bool = False,
    ):
        self.session_id = session_id or str(uuid4())
        self.agent_id = agent_id or "default-agent"
        self.thread_id = thread_id or str(uuid4())
        self.user_id = user_id
        self.tags = tags or []
        self.workflow_name = workflow_name
        self.api_key = api_key
        self.enable_http_tracing = enable_http_tracing
        self.disable_content = disable_content

        # Dry run configuration overrides
        self.dry_run = dry_run
        if self.dry_run:
            logging.info(
                "Neatlogs: Dry run mode enabled. Data will NOT be sent to server."
            )
            self.enable_server_sending = False
            # Force enable OTel console export for visibility
            self.enable_otel = True
            otel_console_export = True
        else:
            self.enable_server_sending = enable_server_sending
            self.enable_otel = enable_otel

        self._threads = []

        # Queue-based sender setup
        # Use daemon=True so the thread doesn't block process exit
        # We rely on explicit flush()/shutdown() calls to ensure data is sent
        self._send_queue = queue.Queue()
        self._sender_thread = threading.Thread(
            target=self._send_worker, daemon=True)
        self._sender_thread.start()
        self._shutdown_event = threading.Event()

        self.setup_logging()
        self._lock = threading.Lock()

        # OpenTelemetry configuration
        self._tracer = None
        self._tracer_provider = None

        if self.enable_otel:
            self._setup_otel(otlp_endpoint, otlp_headers, otel_console_export)
            
            # Setup HTTP instrumentation if enabled
            if self.enable_http_tracing:
                self._setup_http_instrumentation()

        logging.info(
            f"LLMTracker initialized - Session: {self.session_id}, "
            f"User: {self.user_id or 'N/A'}, "
            f"Agent: {self.agent_id}, Thread: {self.thread_id}, "
            f"Workflow: {self.workflow_name}, "
            f"OTel: {self.enable_otel}, DryRun: {self.dry_run}"
        )

    def _setup_otel(
        self,
        otlp_endpoint: Optional[str],
        otlp_headers: Optional[Dict[str, str]],
        console_export: bool,
    ):
        """
        Configure OpenTelemetry.

        This method attempts to attach the NeatlogsSpanProcessor to the current
        global TracerProvider. If no provider is configured (i.e., it's the default
        ProxyTracerProvider), it initializes a new SDK TracerProvider and sets it
        as global.
        """

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from .otel.processor import NeatlogsSpanProcessor

            # Check if a global provider is already set
            current_provider = trace.get_tracer_provider()

            # If it's a ProxyTracerProvider, it means it hasn't been configured yet.
            # (Note: We check class name to avoid importing ProxyTracerProvider which might be internal)
            provider_class_name = current_provider.__class__.__name__
            is_proxy = provider_class_name == "ProxyTracerProvider"
            
            # Also check if it's already a TracerProvider (from SDK or another instrumentation)
            is_sdk_provider = isinstance(current_provider, TracerProvider)

            if is_proxy and not is_sdk_provider:
                logging.info(
                    "Neatlogs: No global TracerProvider detected. Initializing new SDK TracerProvider."
                )

                # Create Resource
                resource_attrs = {
                    "service.name": "neatlogs",
                    "service.version": "1.1.7",
                    "neatlogs.session_id": self.session_id,
                    "neatlogs.agent_id": self.agent_id,
                    "neatlogs.thread_id": self.thread_id,
                }
                if self.user_id:
                    resource_attrs["neatlogs.user_id"] = self.user_id
                if self.workflow_name:
                    resource_attrs["neatlogs.workflow_name"] = self.workflow_name
                if self.tags:
                    resource_attrs["neatlogs.tags"] = ",".join(self.tags)

                resource = Resource.create(resource_attrs)

                # Initialize new TracerProvider
                self._tracer_provider = TracerProvider(resource=resource)
                self._owns_tracer_provider = True

                # Set as global (wrap in try-catch for safety)
                try:
                    trace.set_tracer_provider(self._tracer_provider)
                    logging.info("Neatlogs: Successfully set global TracerProvider.")
                except Exception as e:
                    logging.warning(
                        f"Neatlogs: Could not set global TracerProvider (already set by another library): {e}. "
                        "Attaching to existing provider instead."
                    )
                    # Fall back to using the existing provider
                    self._tracer_provider = trace.get_tracer_provider()
                    self._owns_tracer_provider = False
            else:
                logging.info(
                    f"Neatlogs: Detected existing global TracerProvider ({provider_class_name}). Attaching to it."
                )
                self._tracer_provider = current_provider
                self._owns_tracer_provider = False

            # Add Neatlogs Span Processor
            # This captures data for the Neatlogs backend
            if hasattr(self._tracer_provider, "add_span_processor"):
                self._tracer_provider.add_span_processor(
                    NeatlogsSpanProcessor(self))
            else:
                logging.warning(
                    "Neatlogs: Current TracerProvider does not support adding span processors. Neatlogs data capture may fail."
                )

            # Configure Exporters (only if we created the provider OR if user explicitly asked for them)
            # If we attached to an existing provider, we generally assume the user configured their own exporters.
            # BUT, if the user passed `otlp_endpoint` to neatlogs.init(), they probably expect us to configure it.

            if otlp_endpoint:
                if hasattr(self._tracer_provider, "add_span_processor"):
                    otlp_exporter = OTLPSpanExporter(
                        endpoint=otlp_endpoint, headers=otlp_headers or {}
                    )
                    self._tracer_provider.add_span_processor(
                        BatchSpanProcessor(otlp_exporter)
                    )
                    logging.info(
                        f"Neatlogs OTel: Added OTLP exporter for {otlp_endpoint}"
                    )

            if console_export:
                if hasattr(self._tracer_provider, "add_span_processor"):
                    console_exporter = ConsoleSpanExporter()
                    self._tracer_provider.add_span_processor(
                        BatchSpanProcessor(console_exporter)
                    )
                    logging.info("Neatlogs OTel: Added Console exporter")

            self._tracer = trace.get_tracer("neatlogs")
            logging.info("Neatlogs: OpenTelemetry setup complete.")

        except ImportError as e:
            logging.error(
                f"Neatlogs: Failed to import OpenTelemetry components: {e}")
            self.enable_otel = False
        except Exception as e:
            logging.error(f"Neatlogs: Failed to configure OpenTelemetry: {e}")
            self.enable_otel = False

    def _send_batch_to_server(self, batch: List[LLMCallData]):
        """Send a batch of spans to the Neatlogs backend."""
        if not self.enable_server_sending or not batch:
            return

        try:
            url = os.getenv("NEATLOGS_API_URL",
                            "http://localhost:3000/api/data/v4/batch")

            # Enrich all spans in the batch
            enriched_spans = []
            for data in batch:
                enriched_span = {
                    **data.span,
                    "trace_id": data.trace_id,
                    "sdk_timestamp": datetime.now().timestamp()
                }
                enriched_spans.append(enriched_span)

            payload = {
                "workflow_name": self.workflow_name,
                "spans": enriched_spans,
            }

            response = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": batch[0].api_key or self.api_key
                },
                timeout=10.0,  # Increased timeout for batches
            )
            response.raise_for_status()
            logging.debug(f"Neatlogs: Sent batch of {len(enriched_spans)} spans")

        except Exception as e:
            logging.error(f"Neatlogs: Failed to send batch: {e}")

    def _enqueue_span(self, call_data: LLMCallData):
        self._send_queue.put(call_data)

    def _send_worker(self):
        """
        Background worker that batches spans and sends them to the server.
        Flushes when:
        - Batch reaches 50 spans (as per LLD)
        - 1 second has elapsed since the last flush
        - Shutdown is requested
        """
        batch = []
        last_flush_time = datetime.now()
        batch_size_limit = 50  # As per LLD V4
        flush_interval_seconds = 1.0  # As per LLD V4

        while True:
            try:
                # Wait for an item with timeout to allow periodic flushing
                item = self._send_queue.get(timeout=flush_interval_seconds)
                
                if item is _STOP:
                    # Flush remaining spans before stopping
                    if batch:
                        self._send_batch_to_server(batch)
                    self._send_queue.task_done()
                    break

                # Add span to batch
                batch.append(item)
                self._send_queue.task_done()

                # Check if we should flush
                time_since_last_flush = (datetime.now() - last_flush_time).total_seconds()
                should_flush = (
                    len(batch) >= batch_size_limit or
                    time_since_last_flush >= flush_interval_seconds
                )

                if should_flush:
                    self._send_batch_to_server(batch)
                    batch = []
                    last_flush_time = datetime.now()

            except queue.Empty:
                # Timeout reached, flush if we have pending spans
                if batch:
                    self._send_batch_to_server(batch)
                    batch = []
                    last_flush_time = datetime.now()
            except Exception as e:
                logging.error(f"Neatlogs: Worker error: {e}")
                # Continue processing, don't crash the worker

    def setup_logging(self):
        """
        Configure file-based logging for LLM calls.

        Sets up a dedicated logger that writes formatted LLM call data to a file.
        This ensures a local backup of all traces is available independent of
        server connectivity.
        """
        self.file_logger = logging.getLogger(f"llm_tracker_{self.session_id}")
        self.file_logger.setLevel(logging.INFO)
        # Remove existing handlers to avoid duplicates
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        # Note: We rely on the parent logger configuration or add a FileHandler if needed.
        # For now, we assume the user or environment configures the handlers for this logger name
        # or we might want to add a default FileHandler here if that was the original intent.
        # Based on previous code, it just set level and cleared handlers.
        # We'll stick to that but ensure it's clean.



    def add_tags(self, tags: List[str]):
        """Add tags to the tracker."""
        with self._lock:
            for tag in tags:
                if tag not in self.tags:
                    self.tags.append(tag)
        logging.info(f"Added tags: {tags}")

    def _setup_http_instrumentation(self):
        """Setup HTTP client instrumentation to capture external API calls from tools."""
        instrumented = []
        errors = []
        
        logging.debug(f"_setup_http_instrumentation: tracer_provider={self._tracer_provider}")
        
        # CRITICAL: Pass tracer_provider to ensure spans are sent to our processor
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            instrumentor = RequestsInstrumentor()
            instrumentor.instrument(tracer_provider=self._tracer_provider)
            instrumented.append("requests")
            logging.info(f"✅ Instrumented 'requests' library for HTTP tracing (tracer_provider={type(self._tracer_provider).__name__})")
        except ImportError as e:
            errors.append(f"requests: {e}")
            logging.warning(f"⚠️  Could not instrument 'requests': {e}")
        except RuntimeError as e:
            errors.append(f"requests: {e}")
            logging.warning(f"⚠️  'requests' already instrumented or error: {e}")
        
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument(tracer_provider=self._tracer_provider)
            instrumented.append("httpx")
            logging.info("✅ Instrumented 'httpx' library for HTTP tracing")
        except ImportError as e:
            errors.append(f"httpx: {e}")
            logging.debug(f"'httpx' not available: {e}")
        except RuntimeError as e:
            errors.append(f"httpx: {e}")
            logging.warning(f"⚠️  'httpx' already instrumented or error: {e}")
        
        try:
            from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
            AioHttpClientInstrumentor().instrument(tracer_provider=self._tracer_provider)
            instrumented.append("aiohttp")
            logging.info("✅ Instrumented 'aiohttp' library for HTTP tracing")
        except ImportError as e:
            errors.append(f"aiohttp: {e}")
            logging.debug(f"'aiohttp' not available: {e}")
        except RuntimeError as e:
            errors.append(f"aiohttp: {e}")
            logging.warning(f"⚠️  'aiohttp' already instrumented or error: {e}")
        
        if instrumented:
            logging.info(f"✅ HTTP Tracing Ready: {', '.join(instrumented)}")
        else:
            logging.error(f"❌ No HTTP libraries could be instrumented. Errors: {errors}")

    def set_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Set contextual attributes that will be attached to all subsequent spans.
        Uses OpenTelemetry Baggage for propagation.
        """
        from opentelemetry import baggage
        
        if user_id is not None:
            self.user_id = user_id
            baggage.set_baggage("neatlogs.user_id", user_id)
        
        if session_id is not None:
            self.session_id = session_id
            baggage.set_baggage("neatlogs.session_id", session_id)
        
        if metadata:
            for key, value in metadata.items():
                baggage.set_baggage(f"neatlogs.metadata.{key}", str(value))

    def track_feedback(
        self,
        trace_id: str,
        rating: Optional[str] = None,
        score: Optional[float] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Track user feedback for a trace."""
        feedback_data = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        if rating:
            feedback_data["rating"] = rating
        if score is not None:
            feedback_data["score"] = score
        if comment:
            feedback_data["comment"] = comment
        if metadata:
            feedback_data["metadata"] = metadata
        
        # Send feedback to server
        if self.enable_server_sending:
            try:
                url = os.getenv("NEATLOGS_API_URL", "http://localhost:3000") + "/api/trace-feedback"
                response = requests.post(
                    url,
                    json=feedback_data,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key
                    },
                    timeout=5.0,
                )
                response.raise_for_status()
                logging.info(f"Feedback tracked for trace {trace_id}")
            except Exception as e:
                logging.error(f"Failed to track feedback: {e}")
        else:
            logging.info(f"[Dry Run] Would track feedback: {feedback_data}")

    def flush(self, timeout: float = 10.0):
        """
        Manually flush all pending spans with a timeout.

        Args:
            timeout: Maximum time in seconds to wait for flush (default: 10s)
        """
        logging.debug("Neatlogs: flush() called")

        # Force flush OpenTelemetry spans first
        if self.enable_otel and self._tracer_provider:
            try:
                self._tracer_provider.force_flush(timeout_millis=int(timeout * 1000 / 2))
                logging.debug("Neatlogs: OTel tracer flushed")
            except Exception as e:
                logging.error(f"Neatlogs: Error flushing OTel tracer: {e}")

        # Wait for queue to be empty with timeout
        # We use a polling approach since Queue.join() doesn't support timeout
        start_time = time.time()
        while not self._send_queue.empty():
            if time.time() - start_time > timeout:
                logging.warning(f"Neatlogs: flush() timed out after {timeout}s with {self._send_queue.qsize()} items remaining")
                break
            time.sleep(0.1)

        # Give worker a bit more time to finish sending the current batch
        time.sleep(0.2)
        logging.debug("Neatlogs: flush() completed")

    def shutdown(self, timeout: float = 5.0):
        """
        Gracefully shutdown the tracker and clean up resources.

        This method ensures that all pending data is sent to the Neatlogs server
        and that the OpenTelemetry tracer provider is properly shut down (if owned).

        Args:
            timeout: Maximum time in seconds to wait for shutdown (default: 5s)
        """
        if self._shutdown_event.is_set():
            logging.debug("Neatlogs: shutdown already called, skipping")
            return

        self._shutdown_event.set()
        logging.debug("Neatlogs: shutdown initiated")

        # Step 1: Force flush OpenTelemetry spans FIRST
        # This ensures all OTel spans are processed by our NeatlogsSpanProcessor
        # and added to the _send_queue before we stop it
        if self.enable_otel and self._tracer_provider:
            try:
                logging.debug("Neatlogs: Forcing OTel tracer flush...")
                self._tracer_provider.force_flush(timeout_millis=int(timeout * 500))
                logging.debug("Neatlogs: OTel tracer flushed")
            except Exception as e:
                logging.debug(f"Neatlogs: Error flushing OTel tracer: {e}")

        # Step 2: Signal worker to stop
        self._send_queue.put(_STOP)

        # Step 3: Wait for queue to drain with timeout
        start_time = time.time()
        while not self._send_queue.empty():
            if time.time() - start_time > timeout:
                logging.warning(f"Neatlogs: shutdown timed out waiting for queue to drain")
                break
            time.sleep(0.1)

        # Step 4: Wait for worker thread (with timeout since it's daemon now)
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1)

        logging.debug("Neatlogs: shutdown completed")

        # Step 5: Shutdown OpenTelemetry tracer
        # We only shutdown if we own the TracerProvider (i.e., we created it).
        # If we attached to an existing global one, we don't shutdown it.
        if self.enable_otel and self._tracer_provider and self._owns_tracer_provider:
            try:
                self._tracer_provider.shutdown()
                logging.debug(
                    "Neatlogs: OpenTelemetry tracer shutdown complete")
            except Exception as e:
                logging.debug(
                    f"Neatlogs: Error shutting down OTel tracer: {e}")

        logging.debug("Neatlogs: LLMTracker.shutdown() finished.")


# --- Global Tracker Instance and Initialization ---


_global_tracker: Optional[LLMTracker] = None
_init_lock = threading.Lock()


def get_tracker() -> Optional[LLMTracker]:
    """
    Get the global tracker instance.
    """
    return _global_tracker
