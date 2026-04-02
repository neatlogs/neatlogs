"""
Neatlogs span exporter with batching and async export.
"""

import json
import logging
import os
import sys
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict, List

import requests

from .logger import get_logger

logger = get_logger()


class NeatlogsExporter:
    """
    Exports spans to Neatlogs backend with batching for performance.

    Features:
    - Batch export to reduce HTTP overhead
    - Async export in background thread
    - Automatic retry on failure
    - Configurable flush interval
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = "http://localhost:3000/api/data/v4/batch",
        workflow_name: str = "neatlogs-app",
        batch_size: int = 100,
        flush_interval: float = 5.0,
        request_timeout: float = 30.0,  # Increase default to 30 seconds for large payloads
        max_retries: int = 3,
        disable_export: bool = False,
    ):
        """
        Initialize the exporter.

        Args:
            api_key: Neatlogs API key
            endpoint: Neatlogs backend endpoint
            workflow_name: Workflow name included at batch payload top level
            batch_size: Maximum number of spans per batch
            flush_interval: Seconds between automatic flushes
            request_timeout: HTTP request timeout in seconds (default: 30s for large payloads)
            max_retries: Maximum retry attempts on failure
        """
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.workflow_name = (
            workflow_name.strip() if isinstance(workflow_name, str) and workflow_name.strip() else "neatlogs-app"
        )
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        # Default false; can be enabled via param or env for unit tests/local debugging.
        self.disable_export = bool(disable_export) or (
            os.getenv("NEATLOGS_DISABLE_EXPORT", "").lower() in ["true", "1", "yes"]
        )

        self._queue: Queue = Queue()
        self._batch: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        self._metrics_queue: Queue = Queue()
        self._metrics_batch: List[Dict[str, Any]] = []
        self._metrics_lock = threading.Lock()

        self._log_spans_enabled = os.getenv("NEATLOGS_LOG_SPANS", "").lower() in [
            "true",
            "1",
            "yes",
        ]
        self._log_file_path = None
        self._log_file_handle = None

        self._log_metrics_enabled = os.getenv("NEATLOGS_LOG_METRICS", "").lower() in [
            "true",
            "1",
            "yes",
        ]
        self._metrics_log_file_path = None
        self._metrics_log_file_handle = None

        if self._log_spans_enabled:
            self._log_file_path = os.path.join(
                os.getcwd(), os.getenv("NEATLOGS_LOG_SPANS_FILE", "spans_optimized.log")
            )
            try:
                self._log_file_handle = open(self._log_file_path, "a", encoding="utf-8")
                logger.info(f"Span logging enabled: {self._log_file_path}")
            except Exception as e:
                logger.warning(f"Failed to open span log file: {e}")
                self._log_spans_enabled = False

        if self._log_metrics_enabled:
            self._metrics_log_file_path = os.path.join(
                os.getcwd(), os.getenv("NEATLOGS_LOG_METRICS_FILE", "metrics_optimized.log")
            )
            try:
                self._metrics_log_file_handle = open(
                    self._metrics_log_file_path, "a", encoding="utf-8"
                )
                logger.info(f"Metrics logging enabled: {self._metrics_log_file_path}")
            except Exception as e:
                logger.warning(f"Failed to open metrics log file: {e}")
                self._log_metrics_enabled = False

        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()

    def export(self, span_data: Dict[str, Any]) -> None:
        """
        Add a span to the export queue.

        Args:
            span_data: Span data dictionary
        """
        if self._stop_event.is_set():
            return

        if self._log_spans_enabled and self._log_file_handle and not self._log_file_handle.closed:
            try:
                json_line = json.dumps(span_data) + "\n"
                self._log_file_handle.write(json_line)
                self._log_file_handle.flush()
            except Exception as e:
                logger.debug(f"Failed to log span to file: {e}")

        self._queue.put(span_data)

        with self._lock:
            with self._metrics_lock:
                if len(self._batch) >= self.batch_size:
                    self._flush_combined_batch()

    def export_metrics(self, metrics_list: List[Dict[str, Any]]) -> None:
        """
        Add metrics to the export queue.

        Args:
            metrics_list: List of metric data point dictionaries
        """
        if self._stop_event.is_set():
            return

        if (
            self._log_metrics_enabled
            and self._metrics_log_file_handle
            and not self._metrics_log_file_handle.closed
        ):
            try:
                for metric_data in metrics_list:
                    json_line = json.dumps(metric_data) + "\n"
                    self._metrics_log_file_handle.write(json_line)
                self._metrics_log_file_handle.flush()
            except Exception as e:
                logger.debug(f"Failed to log metrics to file: {e}")

        for metric_data in metrics_list:
            self._metrics_queue.put(metric_data)

        with self._lock:
            with self._metrics_lock:
                if len(self._metrics_batch) >= self.batch_size:
                    self._flush_combined_batch()

    def flush(self, timeout: float = 10.0) -> None:
        """
        Force flush all pending spans and metrics.

        This method blocks until all queued items are processed and sent to the backend.

        Args:
            timeout: Maximum time to wait for flush (seconds)
        """
        spans_collected = 0
        while not self._queue.empty():
            try:
                span_data = self._queue.get_nowait()
                with self._lock:
                    self._batch.append(span_data)
                spans_collected += 1
            except Empty:
                break

        metrics_collected = 0
        while not self._metrics_queue.empty():
            try:
                metric_data = self._metrics_queue.get_nowait()
                with self._metrics_lock:
                    self._metrics_batch.append(metric_data)
                metrics_collected += 1
            except Empty:
                break

        with self._lock:
            with self._metrics_lock:
                if self._batch or self._metrics_batch:
                    self._flush_combined_batch()

        for _ in range(spans_collected):
            self._queue.task_done()
        for _ in range(metrics_collected):
            self._metrics_queue.task_done()

        try:
            logger.debug("Waiting for span queue to drain...")
            self._queue.join()
            logger.debug("Waiting for metrics queue to drain...")
            self._metrics_queue.join()
            logger.debug("All queues drained successfully")
        except Exception as e:
            logger.warning(f"Error waiting for queues to drain: {e}")

    def shutdown(self) -> None:
        """
        Shutdown the exporter and flush all pending spans.
        """
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True

        self._stop_event.set()
        self.flush()
        self._flush_thread.join(timeout=10.0)

        if self._log_file_handle:
            try:
                self._log_file_handle.close()
                logger.debug(f"Span log file closed: {self._log_file_path}")
            except Exception:
                pass
            finally:
                self._log_file_handle = None

        if self._metrics_log_file_handle:
            try:
                self._metrics_log_file_handle.close()
                logger.debug(f"Metrics log file closed: {self._metrics_log_file_path}")
            except Exception:
                pass
            finally:
                self._metrics_log_file_handle = None

    def _flush_worker(self) -> None:
        """
        Background worker that periodically flushes spans and metrics batches.
        """
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)

            spans_collected = 0
            spans_to_mark_done = 0
            while not self._queue.empty() and spans_collected < self.batch_size:
                try:
                    span_data = self._queue.get_nowait()
                    with self._lock:
                        self._batch.append(span_data)
                    spans_collected += 1
                    spans_to_mark_done += 1
                except Empty:
                    break

            metrics_collected = 0
            metrics_to_mark_done = 0
            while not self._metrics_queue.empty() and metrics_collected < self.batch_size:
                try:
                    metric_data = self._metrics_queue.get_nowait()
                    with self._metrics_lock:
                        self._metrics_batch.append(metric_data)
                    metrics_collected += 1
                    metrics_to_mark_done += 1
                except Empty:
                    break

            with self._lock:
                with self._metrics_lock:
                    if self._batch or self._metrics_batch:
                        self._flush_combined_batch()

            for _ in range(spans_to_mark_done):
                self._queue.task_done()
            for _ in range(metrics_to_mark_done):
                self._metrics_queue.task_done()

    def _flush_combined_batch(self) -> None:
        """
        Flush both spans and metrics in a single request to Neatlogs backend.

        NOTE: Must be called while holding both self._lock and self._metrics_lock
        """
        if not self._batch and not self._metrics_batch:
            return

        spans_to_send = self._batch.copy() if self._batch else []
        metrics_to_send = self._metrics_batch.copy() if self._metrics_batch else []

        self._batch.clear()
        self._metrics_batch.clear()

        if self.disable_export:
            logger.debug(
                f"Export disabled: skipping send of {len(spans_to_send)} spans and {len(metrics_to_send)} metrics"
            )
            return

        payload = {}
        payload["workflow_name"] = self.workflow_name
        if spans_to_send:
            payload["spans"] = spans_to_send
        if metrics_to_send:
            payload["metrics"] = metrics_to_send

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.request_timeout,
                )

                if response.status_code == 200:
                    logger.debug(
                        f"Successfully exported {len(spans_to_send)} spans and {len(metrics_to_send)} metrics"
                    )
                    break
                elif response.status_code == 401:
                    self._emit_export_error(
                        "Authentication failed (invalid API key)"
                    )
                    break
                elif response.status_code == 429:
                    self._emit_export_error(
                        f"Export rate-limited (429): {self._response_snippet(response)}"
                    )
                    break
                elif response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Server error {response.status_code}, retrying (attempt {attempt + 1}/{self.max_retries})..."
                        )
                        time.sleep(2**attempt)
                        continue
                    else:
                        self._emit_export_error(
                            f"Server error {response.status_code} after {self.max_retries} attempts, giving up"
                        )
                else:
                    self._emit_export_error(
                        f"Export failed with status {response.status_code}: {self._response_snippet(response)}"
                    )
                    break

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request exception, retrying (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    time.sleep(2**attempt)
                    continue
                else:
                    self._emit_export_error(
                        f"Export failed after {self.max_retries} attempts: {e}"
                    )

    def _response_snippet(self, response: requests.Response, max_len: int = 240) -> str:
        """
        Return a short, safe snippet of an HTTP response body for error logs.
        """
        try:
            body = (response.text or "").strip().replace("\n", " ")
            if not body:
                return "(empty response body)"
            if len(body) > max_len:
                return body[:max_len] + "..."
            return body
        except Exception:
            return "(unable to read response body)"

    def _emit_export_error(self, message: str) -> None:
        """
        Emit exporter errors even when debug=False.
        Falls back to stderr if logger error level is disabled.
        """
        try:
            logger.error(message)
            if logger.disabled or not logger.isEnabledFor(logging.ERROR):
                print(f"[neatlogs] ERROR: {message}", file=sys.stderr, flush=True)
        except Exception:
            print(f"[neatlogs] ERROR: {message}", file=sys.stderr, flush=True)
