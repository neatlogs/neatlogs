#!/usr/bin/env python3
"""Emit a deterministic user workflow through the installed NeatLogs package."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any

from opentelemetry.trace import Status, StatusCode

import neatlogs


def _identity(span: Any) -> tuple[str, str]:
    context = span.get_span_context()
    return f"{context.trace_id:032x}", f"{context.span_id:016x}"


def _finish(span: Any, input_value: Any, output_value: Any) -> None:
    span.set_attribute("neatlogs.internal", False)
    span.set_attribute("input.value", json.dumps(input_value, sort_keys=True))
    span.set_attribute("output.value", json.dumps(output_value, sort_keys=True))
    span.set_status(Status(StatusCode.OK))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("NEATLOGS_API_KEY")
    endpoint = os.environ.get("NEATLOGS_ENDPOINT", "https://ingest.neatlogs.com")
    if not api_key and not args.local:
        raise SystemExit("NEATLOGS_API_KEY is required")

    run_id = os.environ.get("NEATLOGS_E2E_RUN_ID") or secrets.token_hex(8)
    prefix = f"python.packaged-e2e.{run_id}"
    expected: list[dict[str, str | None]] = []

    neatlogs.init(
        api_key=api_key or "local-e2e-placeholder",
        endpoint=endpoint,
        workflow_name=prefix,
        instrumentations=[],
        batch_size=2,
        flush_interval=0.25,
        disable_export=args.local,
        register_shutdown_handlers=False,
    )

    with neatlogs.trace(
        name=f"{prefix}.root",
        kind="WORKFLOW",
        session_id=f"session-{run_id}",
        end_user_id=f"user-{run_id}",
        **{"neatlogs.verification.marker": run_id},
    ) as root:
        trace_id, root_id = _identity(root)
        expected.append(
            {"name": f"{prefix}.root", "kind": "workflow", "span_id": root_id, "parent": None}
        )

        with neatlogs.trace(name=f"{prefix}.agent", kind="AGENT") as agent:
            _, agent_id = _identity(agent)
            expected.append(
                {
                    "name": f"{prefix}.agent",
                    "kind": "agent_action",
                    "span_id": agent_id,
                    "parent": root_id,
                }
            )
            with neatlogs.trace(name=f"{prefix}.llm", kind="LLM") as llm:
                _, llm_id = _identity(llm)
                expected.append(
                    {
                        "name": f"{prefix}.llm",
                        "display_name": "e2e-deterministic-model",
                        "kind": "llm",
                        "span_id": llm_id,
                        "parent": agent_id,
                    }
                )
                llm.set_attribute("llm.model_name", "e2e-deterministic-model")
                llm.set_attribute("neatlogs.llm.token_count.prompt", 13)
                llm.set_attribute("neatlogs.llm.token_count.completion", 5)
                llm.set_attribute("neatlogs.llm.token_count.total", 18)
                _finish(
                    llm,
                    {"messages": [{"role": "user", "content": "deterministic launch check"}]},
                    {"text": "deterministic response"},
                )
            _finish(agent, {"task": "answer"}, {"answer": "deterministic response"})

        with neatlogs.trace(name=f"{prefix}.retriever", kind="RETRIEVER") as retriever:
            _, retriever_id = _identity(retriever)
            expected.append(
                {
                    "name": f"{prefix}.retriever",
                    "kind": "retriever",
                    "span_id": retriever_id,
                    "parent": root_id,
                }
            )
            _finish(retriever, {"query": "launch readiness"}, {"documents": ["fixture-doc"]})

        with neatlogs.trace(name=f"{prefix}.tool", kind="TOOL") as tool:
            _, tool_id = _identity(tool)
            expected.append(
                {
                    "name": f"{prefix}.tool",
                    "kind": "tool_call",
                    "span_id": tool_id,
                    "parent": root_id,
                }
            )
            _finish(tool, {"value": 2}, {"value": 4})

        _finish(root, {"request": "launch readiness"}, {"status": "ok"})

    try:
        flushed = neatlogs.flush(timeout_millis=10_000)
    finally:
        neatlogs.shutdown(timeout_millis=10_000, termination_reason="packaged-e2e-complete")

    manifest = {
        "format_version": "neatlogs.packaged-e2e/v1",
        "run_id": run_id,
        "trace_id": trace_id,
        "expected_span_count": len(expected),
        "expected_prompt_tokens": 13,
        "expected_completion_tokens": 5,
        "expected_total_tokens": 18,
        "flush_success": bool(flushed),
        "spans": expected,
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trace_id": trace_id, "span_count": len(expected), "flush": bool(flushed)}))
    return 0 if flushed else 1


if __name__ == "__main__":
    raise SystemExit(main())
