"""
Explicit prompt capture helpers.
"""

import json
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.context import attach, detach, get_current, set_value


def capture_prompt(
    template: str,
    variables: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
) -> None:
    current_span = trace.get_current_span()

    if current_span and current_span.is_recording():
        current_span.set_attribute("llm.prompt_template", template)
        if variables:
            current_span.set_attribute("llm.prompt_template_variables", json.dumps(variables))

        if version:
            current_span.set_attribute("llm.prompt_template.version", version)

        ctx = get_current()
        ctx = set_value("neatlogs.prompt_template", template, context=ctx)
        if variables:
            ctx = set_value(
                "neatlogs.prompt_variables", json.dumps(variables, default=str), context=ctx
            )
        if version:
            ctx = set_value("neatlogs.prompt_version", version, context=ctx)
        attach(ctx)


def capture_vars(**kwargs) -> None:
    current_span = trace.get_current_span()

    if current_span and current_span.is_recording():
        current_span.set_attribute("llm.prompt_template_variables", json.dumps(kwargs))
