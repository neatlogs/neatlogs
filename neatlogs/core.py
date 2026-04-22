"""
Core tracking functionality for Neatlogs Tracker
"""

import time
import json
import threading
import logging
import traceback
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests

import contextvars

# Context variable for agentic framework
_current_framework_ctx = contextvars.ContextVar(
    'current_framework', default=None)

# Context variable for parent span
current_span_id_context = contextvars.ContextVar('current_span_id', default=None)


# Context variable to suppress low-level patching
_suppress_patching_ctx = contextvars.ContextVar(
    'suppress_patching', default=False)


def set_current_framework(framework: str):
    """Set the current framework context for the current async task."""
    _current_framework_ctx.set(framework)


def get_current_framework() -> Optional[str]:
    """Get the current framework from the current async task."""
    return _current_framework_ctx.get()


def clear_current_framework():
    """Clear the current framework context."""
    _current_framework_ctx.set(None)


def suppress_patching():
    """Sets a flag to suppress low-level patching for the current async task."""
    _suppress_patching_ctx.set(True)


def release_patching():
    """Releases the suppression flag for low-level patching."""
    _suppress_patching_ctx.set(False)


def is_patching_suppressed() -> bool:
    """Checks if low-level patching is currently suppressed."""
    return _suppress_patching_ctx.get()


# Context variable for passing LangGraph node spans to provider handlers
_active_langgraph_node_span_ctx = contextvars.ContextVar(
    'active_langgraph_node_span', default=None)


def set_active_langgraph_node_span(span: 'LLMSpan'):
    """Set the active LangGraph node span in the current context."""
    _active_langgraph_node_span_ctx.set(span)


def get_active_langgraph_node_span() -> Optional['LLMSpan']:
    """Get the active LangGraph node span from the current context."""
    return _active_langgraph_node_span_ctx.get()


def clear_active_langgraph_node_span():
    """Clear the active LangGraph node span from the current context."""
    _active_langgraph_node_span_ctx.set(None)


@dataclass
class LLMCallData:
    """Data structure for LLM call information"""
    session_id: str
    agent_id: str
    thread_id: str
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    node_type: str
    node_name: str
    model: str
    provider: str
    framework: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    messages: List[Dict]
    completion: str
    timestamp: str
    start_time: float
    end_time: float
    duration: float
    tags: List[str]
    error_report: Optional[Dict] = None
    status: str = "SUCCESS"
    api_key: Optional[str] = None


class LLMSpan:
    """
    Represents a single LLM operation span.

    An LLMSpan tracks a single LLM operation from start to finish, collecting
    metadata, timing information, and token usage. It serves as the primary
    data structure for capturing LLM call details.
    """

    def __init__(self, session_id, agent_id, thread_id, api_key, model=None, provider=None, framework=None, tags=None, node_type: str = "llm_call", node_name: str = None):
        self.span_id = str(uuid4())
        self.parent_span_id = current_span_id_context.get()
        self.trace_id = thread_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.model = model
        self.provider = provider
        self.framework = framework
        self.tags = tags or []
        self.api_key = api_key
        self.messages = []
        self.completion = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cost = 0.0
        self.error_report = None
        self.status = "SUCCESS"
        self.start_time = None
        self.end_time = None
        self.node_type = node_type
        self.node_name = node_name or model or "Unknown Node" 

    def start(self, parent_context=None):
        """Start the span timer"""
        self.start_time = time.time()

    def end(self, success=True, error=None):
        """End the span and record results"""
        self.end_time = time.time()
        self.status = "SUCCESS" if success else "FAILURE"
        if error:
            self.error_report = {
                "error_type": type(error).__name__,
                "error_code": getattr(error, 'code', 'N/A'),
                "error_message": str(error),
                "stack_trace": traceback.format_exc()
            }

    def to_llm_call_data(self) -> LLMCallData:
        """Convert span to LLM call data"""
        duration = (
            self.end_time - self.start_time) if self.end_time and self.start_time else 0
        return LLMCallData(
            session_id=self.session_id,
            agent_id=self.agent_id,
            thread_id=self.thread_id,
            span_id=self.span_id,
            trace_id=self.trace_id,
            parent_span_id=self.parent_span_id,
            node_type=self.node_type,
            node_name=self.node_name,
            model=self.model or "unknown",
            provider=self.provider or "unknown",
            framework=self.framework,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cost=self.cost,
            messages=self.messages,
            completion=self.completion,
            timestamp=datetime.fromtimestamp(
                self.start_time).isoformat() if self.start_time else datetime.now().isoformat(),
            start_time=self.start_time,
            end_time=self.end_time,
            duration=duration,
            tags=self.tags,
            error_report=self.error_report,
            status=self.status,
            api_key=self.api_key
        )


class LLMTracker:
    """
    Main orchestrator for LLM tracking, logging, and reporting.
    The LLMTracker manages the lifecycle of LLM operations, from span creation
    to data collection and reporting. It handles both file-based logging and
    server-side telemetry transmission.
    Key Responsibilities:
    - Managing active spans and completed calls
    - Coordinating background threads for server communication
    - Handling graceful shutdown procedures
    - Providing thread-safe operations for concurrent environments
    """

    def __init__(self, api_key, session_id=None, agent_id=None, thread_id=None, tags=None, enable_server_sending=True):
        self.session_id = session_id or str(uuid4())
        self.agent_id = agent_id or "default-agent"
        self.thread_id = thread_id or str(uuid4())
        self.tags = tags or []
        self.api_key = api_key
        self.enable_server_sending = enable_server_sending
        self._threads = []

        self.setup_logging()
        self._lock = threading.Lock()
        self._active_spans = {}
        self._completed_calls = []

        logging.info(f"LLMTracker initialized - Session: {self.session_id}, "
                     f"Agent: {self.agent_id}, Thread: {self.thread_id}")

    def _send_data_to_server(self, call_data: LLMCallData):
        """
        Send trace data to Neatlogs server in a background thread.
        This method handles the transmission of collected telemetry data to the
        Neatlogs backend service. It includes error handling and basic retry logic
        to ensure data reliability.
        Args:
            call_data (LLMCallData): The serialized LLM call data to transmit
        """
        def send_in_background():
            try:
                url = "https://app.neatlogs.com/api/data/v2"
                headers = {"Content-Type": "application/json"}
                trace_data = asdict(call_data)
                api_data = {
                    "dataDump": json.dumps(trace_data),
                    "projectAPIKey": call_data.api_key or self.api_key,
                    "externalTraceId": call_data.trace_id,
                    "timestamp": datetime.now().timestamp()
                }
                logging.debug(f"Neatlogs: Sending data to server at {url}")
                response = requests.post(
                    url, json=api_data, headers=headers, timeout=10.0)
                response.raise_for_status()
                logging.debug(
                    f"Neatlogs: Successfully sent data to server, status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error sending data to server: {e}")
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred in send_in_background: {e}")

        thread = threading.Thread(target=send_in_background, daemon=False)
        thread.start()
        self._threads.append(thread)

    def setup_logging(self):
        """
        Setup file-based logging with proper formatting.
        This method configures a dedicated logger for this tracker instance,
        ensuring that LLM call data is properly formatted and written to log files.
        It removes any existing handlers to prevent duplicate logs.
        """
        self.file_logger = logging.getLogger(f'llm_tracker_{self.session_id}')
        self.file_logger.setLevel(logging.INFO)
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

    def start_llm_span(self, model=None, provider=None, framework=None, node_type: str = "llm_call", node_name: str = None) -> 'LLMSpan':
        """
        Create and start a new LLM span for tracking an operation.
        This method initializes a new LLMSpan with the provided parameters and
        starts its timer. The span is registered in the active spans dictionary
        for later retrieval and completion.
        Args:
            model (str, optional): The LLM model name
            provider (str, optional): The LLM provider name
            framework (str, optional): The agentic framework being used
            node_type (str, optional): The type of node being tracked.
            node_name (str, optional): A human-readable name for the node.
        Returns:
            LLMSpan: The newly created and started span
        """
        _framework = framework if framework is not None else (
            get_current_framework() or None)
        span = LLMSpan(self.session_id, self.agent_id, self.thread_id,
                       self.api_key, model, provider, _framework, self.tags, node_type=node_type, node_name=node_name)
        with self._lock:
            self._active_spans[span.span_id] = span
        span.start()
        return span

    def end_llm_span(self, span, success=True, error=None):
        """
        Complete an LLM span and log the call data.
        This method finalizes an LLMSpan, converts it to LLMCallData, and logs
        the information both to file and potentially to the Neatlogs server.
        Args:
            span (LLMSpan): The span to complete
            success (bool): Whether the operation was successful
            error (Exception, optional): Error if operation failed
        """
        span.end(success, error)
        with self._lock:
            if span.span_id in self._active_spans:
                del self._active_spans[span.span_id]
            call_data = span.to_llm_call_data()
            self._completed_calls.append(call_data)
            self.log_llm_call(call_data)

    def log_llm_call(self, call_data: LLMCallData):
        log_entry = {"event_type": "LLM_CALL", "data": asdict(call_data)}
        self.file_logger.info(json.dumps(log_entry, indent=2))
        if self.enable_server_sending:
            logging.debug(
                "Neatlogs: Creating background thread to send call_data to server")
            self._send_data_to_server(call_data)

    def get_session_stats(self) -> Dict:
        """
        Get statistics for all completed LLM calls in this session.

        Returns:
            Dict: Statistics including total_calls, tokens, cost, and breakdowns
        """
        with self._lock:
            if not self._completed_calls:
                return {
                    "total_calls": 0,
                    "total_tokens_input": 0,
                    "total_tokens_output": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "average_response_time": 0.0,
                    "provider_breakdown": {},
                    "model_breakdown": {},
                    "status_breakdown": {},
                }

            total_calls = len(self._completed_calls)
            total_prompt_tokens = sum(c.prompt_tokens for c in self._completed_calls)
            total_completion_tokens = sum(c.completion_tokens for c in self._completed_calls)
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost = sum(c.cost for c in self._completed_calls)

            response_times = [
                c.duration for c in self._completed_calls if c.duration > 0
            ]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

            provider_breakdown = {}
            for c in self._completed_calls:
                provider = c.provider or "unknown"
                if provider not in provider_breakdown:
                    provider_breakdown[provider] = {
                        "calls": 0, "cost": 0.0, "tokens": 0
                    }
                provider_breakdown[provider]["calls"] += 1
                provider_breakdown[provider]["cost"] += c.cost
                provider_breakdown[provider]["tokens"] += c.total_tokens

            model_breakdown = {}
            for c in self._completed_calls:
                model = c.model or "unknown"
                if model not in model_breakdown:
                    model_breakdown[model] = {
                        "calls": 0, "cost": 0.0, "tokens": 0
                    }
                model_breakdown[model]["calls"] += 1
                model_breakdown[model]["cost"] += c.cost
                model_breakdown[model]["tokens"] += c.total_tokens

            status_breakdown = {}
            for c in self._completed_calls:
                status = c.status or "unknown"
                if status not in status_breakdown:
                    status_breakdown[status] = 0
                status_breakdown[status] += 1

            return {
                "total_calls": total_calls,
                "total_tokens_input": total_prompt_tokens,
                "total_tokens_output": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 6),
                "average_response_time": round(avg_response_time, 3),
                "provider_breakdown": provider_breakdown,
                "model_breakdown": model_breakdown,
                "status_breakdown": status_breakdown,
            }

    def add_tags(self, tags: List[str]):
        """Add tags to the tracker."""
        with self._lock:
            for tag in tags:
                if tag not in self.tags:
                    self.tags.append(tag)
        logging.info(f"Added tags: {tags}")

    def shutdown(self):
        """Graceful shutdown with proper cleanup"""
        logging.debug(
            f"Neatlogs: LLMTracker.shutdown() called. Waiting for {len(self._threads)} sender threads to complete.")
        for thread in self._threads:
            thread.join(timeout=5.0)
        logging.debug("Neatlogs: LLMTracker.shutdown() finished.")

# --- Global Tracker Instance and Initialization ---


_global_tracker: Optional[LLMTracker] = None
_init_lock = threading.Lock()


def get_tracker() -> Optional[LLMTracker]:
    """
    Get the global tracker instance.
    """
    return _global_tracker
