"""
Neatlogs Strands Agents integration.

Usage:
    >>> import neatlogs
    >>> from strands import Agent
    >>> neatlogs.init(api_key="...", workflow_name="...")   # registers the OTel tracer
    >>> agent = neatlogs.strands_hooks(Agent(model=model))  # installs the I/O hook
    >>> response = agent("Hello")

The Strands Agents SDK emits its OWN OpenTelemetry spans (invoke_agent,
execute_tool, model `chat` calls) as soon as a global tracer provider exists —
which ``neatlogs.init()`` registers. Those spans flow into neatlogs automatically
and the attribute mapper classifies them as AGENT / TOOL / LLM, with token usage
from the gen_ai.usage.* attributes.

However, Strands puts prompt/response CONTENT on span EVENTS (gen_ai.user.message,
gen_ai.system.message, gen_ai.choice, …) per the OTel GenAI convention — NOT on
span attributes. neatlogs renders I/O from span attributes, so without help those
native spans would show tokens but no input/output. ``strands_hooks()`` installs a
single class-level hook on Strands' own ``Tracer._add_event`` chokepoint: whenever
Strands records a gen_ai message/choice event, we ALSO set input.value/output.value
(+ span kind) on the same span. We do NOT create our own spans — Strands' native
tracing stays the source of truth; we only enrich it.
"""

from typing import Any

from ._wrap_utils import serialize

_HOOK_INSTALLED = False


def strands_hooks(agent: Any) -> Any:
    """
    Install the Strands telemetry I/O hook (idempotent, class-level) and return the
    agent unchanged. Strands self-instruments via native OTel; this hook only adds
    input/output content (which Strands keeps on span events) to the span attributes
    neatlogs renders.
    """
    _install_event_hook()
    try:
        setattr(agent, "_neatlogs_patched", True)
    except Exception:
        pass
    return agent


def _install_event_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    try:
        from strands.telemetry.tracer import Tracer
    except Exception:
        return

    orig_add_event = getattr(Tracer, "_add_event", None)
    if orig_add_event is None or getattr(orig_add_event, "_neatlogs_wrapped", False):
        _HOOK_INSTALLED = True
        return

    def patched_add_event(self, span, event_name, event_attributes=None, *args, **kwargs):
        orig_add_event(self, span, event_name, event_attributes, *args, **kwargs)
        try:
            if span is None or not getattr(span, "is_recording", lambda: False)():
                return
            ev_attrs = event_attributes or {}
            # Classify the span from strands' own gen_ai.operation.name (set at span
            # creation): chat → llm, execute_tool → tool, invoke_agent → agent, else
            # chain. We DON'T set neatlogs.span.kind (the mapper does that from the
            # span name); we only use the op to write I/O under the RIGHT namespace,
            # because the generic {span_kind}.input mapping can't resolve the kind in
            # time for these natively-created spans.
            op = ""
            try:
                op = str((span.attributes or {}).get("gen_ai.operation.name", "")).lower()
            except Exception:
                op = ""
            is_tool = op == "execute_tool" or "gen_ai.tool.name" in (getattr(span, "attributes", {}) or {})
            in_key = "neatlogs.tool.input" if is_tool else None
            out_key = "neatlogs.tool.output" if is_tool else None

            # Input-side messages: gen_ai.{system,user,assistant,tool}.message
            if event_name.startswith("gen_ai.") and event_name.endswith(".message"):
                role = event_name[len("gen_ai."):-len(".message")]
                content = _strands_event_text(ev_attrs.get("content"))
                if content:
                    if is_tool:
                        span.set_attribute("input.value", content)
                        span.set_attribute("neatlogs.tool.input", content)
                    else:
                        _append_input_message(span, role, content)
            # Output: gen_ai.choice (legacy) or the new latest-convention details event.
            elif event_name in ("gen_ai.choice", "gen_ai.client.inference.operation.details"):
                key = "message" if event_name == "gen_ai.choice" else "gen_ai.output.messages"
                out = _strands_event_text(ev_attrs.get(key))
                if out:
                    span.set_attribute("output.value", out)
                    if is_tool:
                        span.set_attribute("neatlogs.tool.output", out)
                    else:
                        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                        span.set_attribute("neatlogs.llm.output_messages.0.content", out)
                        span.set_attribute("neatlogs.llm.output", serialize({"role": "assistant", "content": out}))
        except Exception:
            pass

    patched_add_event._neatlogs_wrapped = True
    Tracer._add_event = patched_add_event
    _HOOK_INSTALLED = True


# Per-span running index for input messages (kept on the span object itself so
# concurrent spans don't collide).
def _append_input_message(span: Any, role: str, content: str) -> None:
    idx = getattr(span, "_neatlogs_in_idx", 0)
    span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
    span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", content)
    try:
        setattr(span, "_neatlogs_in_idx", idx + 1)
    except Exception:
        pass
    # Maintain a flat input.value blob (latest wins; cheap to overwrite).
    existing = getattr(span, "_neatlogs_in_msgs", [])
    existing.append({"role": role, "content": content})
    try:
        setattr(span, "_neatlogs_in_msgs", existing)
    except Exception:
        pass
    blob = serialize({"messages": existing})
    span.set_attribute("input.value", blob)
    # Flat LLM input the backend reads directly (namespace-correct without relying
    # on the {span_kind} mapping, which can't resolve the kind for native spans).
    span.set_attribute("neatlogs.llm.input", blob)


def _strands_event_text(content: Any) -> str:
    """
    Flatten Strands/Bedrock content into readable text. Content arrives as a JSON
    string or list of blocks: [{"text": "..."}], [{"toolUse": {...}}],
    [{"toolResult": {...}}], or [{"role","parts","finish_reason"}].
    """
    if content is None:
        return ""
    val = content
    if isinstance(val, str):
        s = val.strip()
        if not (s.startswith("[") or s.startswith("{")):
            return val  # already plain text
        try:
            import json

            val = json.loads(s)
        except Exception:
            return val
    return _flatten_blocks(val)


def _flatten_blocks(val: Any) -> str:
    out = []
    items = val if isinstance(val, list) else [val]
    for item in items:
        if isinstance(item, str):
            out.append(item)
            continue
        if not isinstance(item, dict):
            out.append(str(item))
            continue
        if "text" in item:
            out.append(str(item["text"]))
        elif "toolUse" in item:
            tu = item["toolUse"] or {}
            out.append(f"{tu.get('name', 'tool')}({serialize(tu.get('input', {}))})")
        elif "toolResult" in item:
            tr = item["toolResult"] or {}
            out.append(_flatten_blocks(tr.get("content", tr)))
        elif "parts" in item:
            out.append(_flatten_blocks(item["parts"]))
        elif "content" in item:
            out.append(_flatten_blocks(item["content"]))
        else:
            out.append(serialize(item))
    return "\n".join(s for s in out if s)
