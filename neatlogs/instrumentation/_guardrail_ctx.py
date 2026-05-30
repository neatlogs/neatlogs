"""Shared contextvar carrying the text a guardrail is checking.

The OpenAI Agents SDK's GuardrailSpanData only exposes {name, triggered}; the checked
input/output is available inside the guardrail runner functions but never written to the
span. The manager patches run_single_input/output_guardrail to set this contextvar, and
the OI openai_agents processor reads it in on_span_end to emit neatlogs.guardrail.input.
"""

import contextvars

GUARDRAIL_INPUT_VAR: "contextvars.ContextVar[str | None]" = contextvars.ContextVar(
    "neatlogs_guardrail_input", default=None
)
