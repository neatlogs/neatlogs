"""
Neatlogs span exporter with batching and async export.
"""

import json
import os
import time
import threading
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
import requests


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
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_retries: int = 3,
    ):
        """
        Initialize the exporter.
        
        Args:
            api_key: Neatlogs API key
            endpoint: Neatlogs backend endpoint
            batch_size: Maximum number of spans per batch
            flush_interval: Seconds between automatic flushes
            max_retries: Maximum retry attempts on failure
        """
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        
        # Span queue for batching
        self._queue: Queue = Queue()
        self._batch: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Check if span logging is enabled via env var
        self._log_spans_enabled = os.getenv("NEATLOGS_LOG_SPANS", "").lower() in ["true", "1", "yes"]
        self._log_file_path = None
        self._log_file_handle = None
        
        if self._log_spans_enabled:
            # Log to spans.log in current working directory
            self._log_file_path = os.path.join(os.getcwd(), "spans_new.log")
            try:
                self._log_file_handle = open(self._log_file_path, 'a', encoding='utf-8')
                print(f"📝 Span logging enabled: {self._log_file_path}")
            except Exception as e:
                print(f"⚠️  Failed to open span log file: {e}")
                self._log_spans_enabled = False
        
        # Background flush thread
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()
    
    def export(self, span_data: Dict[str, Any]) -> None:
        """
        Add a span to the export queue.
        
        Args:
            span_data: Span data dictionary
        """
        # Log span to file if enabled
        if self._log_spans_enabled and self._log_file_handle:
            try:
                # Write full span JSON to file (one JSON object per line)
                json_line = json.dumps(span_data) + '\n'
                self._log_file_handle.write(json_line)
                self._log_file_handle.flush()  # Ensure it's written immediately
            except Exception as e:
                # Don't let logging errors break the export
                print(f"⚠️  Failed to log span: {e}")
        
        self._queue.put(span_data)
        
        # Check if we need to flush immediately
        with self._lock:
            if len(self._batch) >= self.batch_size:
                self._flush_batch()
    
    def flush(self, timeout: float = 10.0) -> None:
        """
        Force flush all pending spans.
        
        Args:
            timeout: Maximum time to wait for flush (seconds)
        """
        # Collect all queued spans
        while not self._queue.empty():
            try:
                span_data = self._queue.get_nowait()
                with self._lock:
                    self._batch.append(span_data)
            except Empty:
                break
        
        # Flush the batch
        with self._lock:
            if self._batch:
                self._flush_batch()
    
    def shutdown(self) -> None:
        """
        Shutdown the exporter and flush all pending spans.
        """
        self._stop_event.set()
        self.flush()
        self._flush_thread.join(timeout=10.0)
        
        # Close log file if open
        if self._log_file_handle:
            try:
                self._log_file_handle.close()
                print(f"📝 Span log file closed: {self._log_file_path}")
            except Exception:
                pass
    
    def _flush_worker(self) -> None:
        """
        Background worker that periodically flushes batches.
        """
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            
            # Collect spans from queue
            spans_collected = 0
            while not self._queue.empty() and spans_collected < self.batch_size:
                try:
                    span_data = self._queue.get_nowait()
                    with self._lock:
                        self._batch.append(span_data)
                    spans_collected += 1
                except Empty:
                    break
            
            # Flush if we have spans
            with self._lock:
                if self._batch:
                    self._flush_batch()
    
    def _flush_batch(self) -> None:
        """
        Flush the current batch to Neatlogs backend.
        
        NOTE: Must be called while holding self._lock
        """
        if not self._batch:
            return
        
        batch_to_send = self._batch.copy()
        self._batch.clear()
        
        # Send batch with retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,  # Use endpoint as-is (already complete URL)
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"spans": batch_to_send},
                    timeout=5.0,  # Shorter timeout to avoid hanging
                )
                
                if response.status_code == 200:
                    # Success
                    break
                elif response.status_code == 401:
                    # Authentication error - don't retry
                    print(f"Neatlogs: Authentication failed (invalid API key)")
                    break
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"Neatlogs: Server error {response.status_code}, giving up")
                else:
                    # Client error - don't retry
                    print(f"Neatlogs: Export failed with status {response.status_code}")
                    break
                    
            except requests.exceptions.RequestException as e:
                # Network error - retry
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Neatlogs: Export failed after {self.max_retries} attempts: {e}")
