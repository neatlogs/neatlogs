"""MCP client that connects to the Neatlogs MCP server and calls log_trace."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("neatlogs.mcp.client")


class MCPTraceClient:
    """Connects to a running Neatlogs MCP server and calls log_trace to send traces.

    Supports two transport modes:
      - Streamable HTTP: connects to an already-running MCP server endpoint
      - SSE: connects via Server-Sent Events (legacy)

    The client maintains a persistent connection and sends traces synchronously
    from the calling thread by dispatching to an internal async event loop.
    """

    def __init__(
        self,
        mcp_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30,
    ):
        self._mcp_url = mcp_url
        self._headers = headers
        self._timeout = timeout
        self._session: Any = None
        self._transport_cm: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._connect_error: Optional[Exception] = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Start background event loop and establish MCP session."""
        with self._lock:
            if self._loop is not None:
                return
            self._connected.clear()
            self._connect_error = None
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._connected.wait(timeout=self._timeout)
            if self._connect_error:
                raise self._connect_error

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_serve())
        except Exception as e:
            self._connect_error = e
            self._connected.set()
        finally:
            self._loop.close()
            self._loop = None

    async def _connect_and_serve(self) -> None:
        from mcp import ClientSession

        try:
            from mcp.client.streamable_http import streamablehttp_client
            transport_factory = streamablehttp_client
        except ImportError:
            from mcp.client.sse import sse_client
            transport_factory = sse_client

        async with transport_factory(
            url=self._mcp_url,
            headers=self._headers,
            timeout=self._timeout,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._session = session
                self._connected.set()
                # Keep alive until close() is called
                try:
                    while True:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    pass
                finally:
                    self._session = None

    def send_trace(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a log_trace call synchronously. Returns the tool result or None on error."""
        if not self._loop or not self._session:
            try:
                self.connect()
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                return None

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._call_log_trace(payload), self._loop
            )
            result = future.result(timeout=self._timeout)
            return result
        except Exception as e:
            logger.error(f"Failed to send trace via MCP: {e}")
            return None

    async def _call_log_trace(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._session:
            return None
        result = await self._session.call_tool("log_trace", payload)
        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        return json.loads(block.text)
                    except (json.JSONDecodeError, TypeError):
                        return {"raw": block.text}
        return None

    def close(self) -> None:
        """Shut down the MCP session and background loop."""
        with self._lock:
            if self._loop and self._loop.is_running():
                for task in asyncio.all_tasks(self._loop):
                    self._loop.call_soon_threadsafe(task.cancel)
            if self._thread:
                self._thread.join(timeout=5)
            self._session = None
            self._loop = None
            self._thread = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
