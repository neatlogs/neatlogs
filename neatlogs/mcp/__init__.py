"""Neatlogs MCP Callbacks — lightweight trace capture via MCP log_trace.

No OTel dependency. Captures spans from LangChain, CrewAI, and direct
provider calls (OpenAI, Anthropic, Google GenAI) into a shared buffer,
then sends complete traces to a Neatlogs MCP server.

Usage::

    from neatlogs.mcp import NeatlogsCallback

    nl = NeatlogsCallback(mcp_url="http://localhost:8080/mcp", workflow_name="my-agent")

    # LangChain / LangGraph
    chain.invoke(input, config={"callbacks": [nl.langchain]})

    # CrewAI
    crew = Crew(
        step_callback=nl.crewai.step,
        task_callback=nl.crewai.task,
        before_kickoff_callbacks=[nl.crewai.before_kickoff],
        after_kickoff_callbacks=[nl.crewai.after_kickoff],
    )

    # Direct provider calls (nest under framework spans via contextvars)
    oai = nl.wrap(openai.OpenAI())
    resp = oai.chat.completions.create(model="gpt-4o", messages=[...])

    # Explicit log spans
    nl.log("Parsed 42 documents from input")

    # Prompt template tracking
    from neatlogs import PromptTemplate
    tpl = PromptTemplate("You are a {{role}} assistant")
    system_msg = tpl.compile(role="expert")  # sets contextvar, auto-captured on next LLM span
"""

from __future__ import annotations

import atexit
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import context as ctx
from .buffer import SpanBuffer, SpanRecord
from .client import MCPTraceClient
from .providers import wrap_client

logger = logging.getLogger("neatlogs.mcp")

__all__ = ["NeatlogsCallback"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any, max_len: int = 50_000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    return s[:max_len] if len(s) > max_len else s


class NeatlogsCallback:
    """Top-level orchestrator for MCP-based trace capture.

    Provides:
      - .langchain   — LangChain/LangGraph BaseCallbackHandler
      - .crewai      — CrewAI adapter (step, task, before_kickoff, after_kickoff)
      - .wrap(client) — proxy for OpenAI/Anthropic/Google GenAI clients
      - .log(msg)     — explicit LOG span
    """

    def __init__(
        self,
        mcp_url: str,
        workflow_name: str = "neatlogs-workflow",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30,
        stale_timeout_s: float = 300.0,
    ):
        self._mcp_url = mcp_url
        self._workflow_name = workflow_name
        self._client = MCPTraceClient(mcp_url, headers=headers, timeout=timeout)
        self._buffer = SpanBuffer(on_flush=self._flush_trace, stale_timeout_s=stale_timeout_s)
        self._langchain_adapter: Any = None
        self._crewai_adapter: Any = None

        atexit.register(self.close)

    def _flush_trace(self, payload: Dict[str, Any]) -> None:
        try:
            result = self._client.send_trace(payload)
            if result:
                logger.debug(f"Trace sent: {result}")
        except Exception as e:
            logger.error(f"Failed to flush trace: {e}")

    # -----------------------------------------------------------------------
    # LangChain / LangGraph
    # -----------------------------------------------------------------------

    @property
    def langchain(self) -> Any:
        """LangChain BaseCallbackHandler instance. Pass to config={"callbacks": [nl.langchain]}."""
        if self._langchain_adapter is None:
            from .langchain import LangChainAdapter
            self._langchain_adapter = LangChainAdapter(
                self._buffer, self._workflow_name, framework="langchain"
            )
        return self._langchain_adapter.handler

    # -----------------------------------------------------------------------
    # CrewAI
    # -----------------------------------------------------------------------

    @property
    def crewai(self) -> Any:
        """CrewAI adapter. Use .step, .task, .before_kickoff, .after_kickoff."""
        if self._crewai_adapter is None:
            from .crewai import CrewAIAdapter
            self._crewai_adapter = CrewAIAdapter(self._buffer, self._workflow_name)
        return self._crewai_adapter

    # -----------------------------------------------------------------------
    # Provider wrapping
    # -----------------------------------------------------------------------

    def wrap(self, client: Any) -> Any:
        """Wrap an OpenAI, Anthropic, or Google GenAI client for auto LLM span capture."""
        return wrap_client(client, self._buffer, self._workflow_name)

    # -----------------------------------------------------------------------
    # Explicit LOG span
    # -----------------------------------------------------------------------

    def log(self, message: str, level: str = "info", **data: Any) -> None:
        """Create a LOG span in the current trace."""
        trace_id = ctx.get_trace_id()
        if not trace_id:
            trace_id = ctx.generate_trace_id()
            ctx.set_trace_id(trace_id)
            self._buffer.get_or_create_trace(trace_id, self._workflow_name)

        parent = ctx.get_parent_span_id()
        rendered = message
        if data:
            rendered = f"{message} | {_safe_json(data)}"

        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "log",
            "neatlogs.log.level": level,
        }

        span = SpanRecord(
            span_id=ctx.generate_span_id(),
            parent_span_id=parent,
            name=message[:80],
            kind="LOG",
            start_time=_now_iso(),
            end_time=_now_iso(),
            status_code="OK",
            attributes=attrs,
        )
        # input = template, output = rendered
        span.attributes["input.value"] = message
        if data:
            span.attributes["output.value"] = _safe_json(data)

        self._buffer.add_span(trace_id, span)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def flush(self) -> None:
        """Flush all pending traces."""
        self._buffer.flush_all()

    def close(self) -> None:
        """Flush and close the MCP connection."""
        try:
            self._buffer.flush_all()
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass
