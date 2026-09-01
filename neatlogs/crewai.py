"""
Neatlogs CrewAI wrapper.

Usage:
    >>> import neatlogs
    >>> from crewai import Crew, Agent, Task
    >>> crew = neatlogs.wrap(Crew(agents=[...], tasks=[...]))
    >>> result = crew.kickoff()

    # Flows are also supported:
    >>> flow = neatlogs.wrap(MyFlow())
    >>> flow.kickoff()

Span hierarchy:
    WORKFLOW (crew.kickoff / Flow.kickoff)
      ↳ TASK   (Task._execute_core / execute_sync / execute_async)
          ↳ AGENT  (Agent.execute_task)
              ↳ LLM   (LLM.call)
              ↳ TOOL  (BaseTool.run)

Crew (WORKFLOW), Task, Agent, TOOL and LLM spans are all installed ONCE at the
class level (Crew.kickoff / Task._execute_core / Agent.execute_task / BaseTool.run
/ LLM.call). So every crew gets a full tree whether or not it was passed through
``neatlogs.wrap()`` — including bare crews under ``instrumentations=["crewai"]``
and mini-crews created deep inside a request. ``wrap()`` only binds metadata.

``kickoff_async`` is NOT patched: CrewAI implements it as
``asyncio.to_thread(self.kickoff, inputs)``, so it delegates to the already-hooked
``kickoff`` on a worker thread; the neatlogs parent key propagates across the
thread via ThreadingInstrumentor, so the async path is covered with no second span.
"""

import json
import os
import sys
import threading
import time
from typing import Any

from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize
from .core.capture import bound_text

_CLASS_HOOKS_INSTALLED = False
_LLM_INPUT_CAPTURE_LIMIT = 1 * 1024 * 1024


def _crewai_span_attributes(kind: str) -> dict[str, Any]:
    return {
        "neatlogs.span.kind": kind,
        "neatlogs.framework": "crewai",
    }


def _serialize_crewai_io(value: Any) -> str:
    """Serialize CrewAI tool I/O within the explicit capture boundary."""
    if isinstance(value, str):
        return bound_text(value)
    try:
        return serialize(value)
    except Exception:
        try:
            return bound_text(str(value))
        except Exception:
            return f"<unserializable {type(value).__name__}>"


def _shared_trace_kwargs() -> dict:
    """Context for the crewai.crew.kickoff root span.

    If a neatlogs span is already active, nest under it (return {}). Otherwise —
    no parent, or only a FOREIGN (non-neatlogs) span active — start as a true root,
    detaching from the foreign parent so the trace has a real root and finalizes.
    """
    from ._wrap_utils import _has_active_recording_parent, _neatlogs_root_kwargs

    if _has_active_recording_parent():
        return {}
    return _neatlogs_root_kwargs()


def _dbg(msg: str) -> None:
    """Emit a debug line only when neatlogs.init(debug=True). Best-effort."""
    try:
        from .init import is_debug_enabled

        if not is_debug_enabled():
            return
        from .core.logger import get_logger

        get_logger().debug(f"[neatlogs.crewai] {msg}")
    except Exception:
        pass


def _caller() -> str:
    """'file:line (in func)' of the user frame that triggered this call."""
    try:
        import inspect

        for fr in inspect.stack()[2:]:
            if "/neatlogs/" not in fr.filename.replace("\\", "/"):
                return f"{fr.filename}:{fr.lineno} (in {fr.function})"
    except Exception:
        pass
    return "<unknown>"


def _suppress_crewai_telemetry() -> None:
    """Disable CrewAI's built-in OTel telemetry.

    CrewAI's telemetry (`crewai.telemetry` scope) emits lifecycle spans
    ("Flow Creation", "Crew Created", "Task Created", …) that carry no I/O. It
    only installs its OWN TracerProvider when none exists; once neatlogs.init()
    has set the global provider, CrewAI's `trace.get_tracer("crewai.telemetry")`
    resolves against OURS, so those noise spans get exported through the neatlogs
    pipeline. Its emission is gated on these env vars (re-read live per op), so
    setting them before kickoff cleanly suppresses the noise. Only set when the
    user hasn't explicitly opted in.

    NOTE: deliberately NOT setting OTEL_SDK_DISABLED — that standard var would
    also disable neatlogs' own OTel exporter. Only the CrewAI-specific vars.

    Two CrewAI telemetry systems exist: the OTel one (crewai.telemetry) disabled
    by CREWAI_DISABLE_TELEMETRY/TRACKING, and the newer "crewai_plus" ephemeral
    tracing that POSTs to app.crewai.com — kept off by CREWAI_TRACING_ENABLED
    staying unset/false (its outbound HTTP would otherwise be traced by our httpx
    instrumentation as a stray second trace).
    """
    for var in ("CREWAI_DISABLE_TELEMETRY", "CREWAI_DISABLE_TRACKING"):
        if os.getenv(var) is None:
            os.environ[var] = "true"
    if os.getenv("CREWAI_TRACING_ENABLED") is None:
        os.environ["CREWAI_TRACING_ENABLED"] = "false"

    # If crewai was imported before us, its Telemetry singleton is already built
    # (env re-read only in __init__), armed with a live exporter to
    # telemetry.crewai.com. Disarm it directly so no lifecycle spans flush.
    tel_mod = sys.modules.get("crewai.telemetry.telemetry")
    inst = getattr(getattr(tel_mod, "Telemetry", None), "_instance", None)
    if inst is not None and getattr(inst, "ready", False):
        inst.ready = False


def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    """
    Set an attribute even on Pydantic models (Crew/Task/Agent are pydantic
    BaseModels that block normal attribute assignment). Falls back to
    object.__setattr__, which bypasses pydantic's __setattr__ validation.
    """
    try:
        setattr(obj, name, value)
    except (ValueError, TypeError, AttributeError):
        try:
            object.__setattr__(obj, name, value)
        except Exception:
            pass


def instrument_crewai() -> None:
    """Install neatlogs' CrewAI class-level hooks (no instance needed).

    Called both by ``neatlogs.wrap(Crew(...))`` and by the instrumentation
    manager for ``instrumentations=["crewai"]``, so bare crews get a full tree.
    Idempotent."""
    _suppress_crewai_telemetry()
    _install_class_hooks()


def wrap_crewai(obj: Any) -> Any:
    """
    Wrap a CrewAI Crew, Flow or standalone Agent instance.

    Span creation is entirely class-level (see :func:`_install_class_hooks`), so
    the instance is already traced the moment the class hooks are installed. This
    only routes Flows / standalone Agents to their dedicated class hooks and
    returns the same instance. Metadata bound at ``wrap(...)`` is applied by the
    proxy in ``neatlogs.wrap``.
    """
    instrument_crewai()

    cls_name = type(obj).__name__
    module = type(obj).__module__ or ""

    obj_id = hex(id(obj))
    crew_name = getattr(obj, "name", None) or getattr(obj, "_name", None)
    _dbg(
        f"wrap_crewai: received {cls_name} (module={module}, id={obj_id}, "
        f"name={crew_name!r}, has_tasks={hasattr(obj, 'tasks')}, "
        f"has_agents={hasattr(obj, 'agents')}) — called from {_caller()}"
    )

    # Flow detection — Flows are their own class hierarchy; patch at the class level.
    if (
        "flow" in module
        or hasattr(obj, "_methods")
        and hasattr(obj, "kickoff")
        and not hasattr(obj, "tasks")
    ):
        _dbg(f"→ routed to FLOW branch; patching flow class {cls_name} (id={obj_id})")
        _patch_flow_class(type(obj))
        return obj

    # Standalone Agent (no Crew): agent.kickoff(messages=...). Covered by the
    # class-level Agent hooks already installed; nothing instance-specific to do.
    if (
        hasattr(obj, "kickoff")
        and hasattr(obj, "execute_task")
        and hasattr(obj, "role")
        and not hasattr(obj, "tasks")
        and not hasattr(obj, "agents")
    ):
        _dbg(
            f"→ routed to STANDALONE AGENT branch (role={getattr(obj, 'role', None)!r}, id={obj_id})"
        )
        _patch_agent_kickoff(obj)
        return obj

    # Crew — fully covered by the class-level Crew.kickoff hook. Nothing per-instance.
    _dbg(
        f"→ CREW branch; class-level hooks cover kickoff* on {cls_name} "
        f"(id={obj_id}, tasks={len(getattr(obj, 'tasks', []) or [])}, "
        f"agents={len(getattr(obj, 'agents', []) or [])})"
    )
    return obj


# ---------------------------------------------------------------------------
# Crew (WORKFLOW spans)
# ---------------------------------------------------------------------------


def _get_crew_attributes(crew: Any) -> dict:
    attrs = _crewai_span_attributes("workflow")

    name = getattr(crew, "name", None) or getattr(crew, "_name", None)
    if name:
        attrs["neatlogs.workflow.name"] = name

    crew_id = getattr(crew, "id", None)
    if crew_id:
        attrs["neatlogs.crewai.crew_id"] = str(crew_id)

    crew_key = getattr(crew, "key", None)
    if crew_key:
        attrs["neatlogs.crewai.crew_key"] = str(crew_key)

    process = getattr(crew, "process", None)
    if process:
        attrs["neatlogs.crewai.process"] = (
            str(process.value) if hasattr(process, "value") else str(process)
        )

    agents = getattr(crew, "agents", None)
    if agents:
        attrs["neatlogs.crewai.crew_number_of_agents"] = len(agents)

    tasks = getattr(crew, "tasks", None)
    if tasks:
        attrs["neatlogs.crewai.crew_number_of_tasks"] = len(tasks)

    try:
        import crewai

        attrs["neatlogs.crewai.version"] = getattr(crewai, "__version__", "")
    except (ImportError, AttributeError):
        pass

    return attrs


def _set_crew_input(span: Any, crew: Any, kwargs: dict) -> None:
    """Set input.value on a crew span.

    Prefer the explicit ``inputs=`` kwarg. When it's absent (bare ``kickoff()``
    with values baked into task descriptions), fall back to the crew's task
    definitions so the workflow root isn't left with an empty input.
    """
    inputs = kwargs.get("inputs")
    if inputs:
        span.set_attribute("input.value", serialize(inputs))
        return
    tasks = getattr(crew, "tasks", None) or []
    derived = []
    for task in tasks:
        desc = getattr(task, "description", "")
        if not desc:
            continue
        entry = {"description": str(desc)}
        expected = getattr(task, "expected_output", "")
        if expected:
            entry["expected_output"] = str(expected)
        derived.append(entry)
    if derived:
        span.set_attribute("input.value", serialize(derived)[:10000])


def _extract_token_usage(result: Any) -> dict:
    attrs = {}
    token_usage = getattr(result, "token_usage", None)
    if not token_usage:
        return attrs
    usage = (
        token_usage
        if isinstance(token_usage, dict)
        else (token_usage.__dict__ if hasattr(token_usage, "__dict__") else {})
    )
    if usage.get("prompt_tokens"):
        attrs["neatlogs.llm.token_count.prompt"] = usage["prompt_tokens"]
    if usage.get("completion_tokens"):
        attrs["neatlogs.llm.token_count.completion"] = usage["completion_tokens"]
    if usage.get("total_tokens"):
        attrs["neatlogs.llm.token_count.total"] = usage["total_tokens"]
    if usage.get("cached_tokens"):
        attrs["neatlogs.llm.token_count.cache_read"] = usage["cached_tokens"]
    return attrs


def _finalize_crew_span(span: Any, result: Any, duration_ms: float) -> None:
    if result is not None:
        raw = getattr(result, "raw", None)
        if raw:
            span.set_attribute("output.value", str(raw)[:10000])
        for attr_name, value in _extract_token_usage(result).items():
            span.set_attribute(attr_name, value)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


# Re-entrancy guard: a crew.kickoff whose body re-enters kickoff on the SAME
# instance in the SAME thread (e.g. CrewAI 1.x's stream=True branch) must not open
# a second WORKFLOW span. kickoff_for_each copies the crew (new id → new span, the
# desired batch→item nesting) and kickoff_async runs on a fresh thread (fresh
# thread-local → span opens), so neither is suppressed. Thread-local so concurrent
# crews on different threads never mask each other.
_kickoff_active = threading.local()


def _kickoff_in_progress(crew: Any) -> bool:
    active = getattr(_kickoff_active, "ids", None)
    return active is not None and id(crew) in active


def _mark_kickoff(crew: Any, on: bool) -> None:
    active = getattr(_kickoff_active, "ids", None)
    if active is None:
        active = set()
        _kickoff_active.ids = active
    if on:
        active.add(id(crew))
    else:
        active.discard(id(crew))


def _patch_crew_class(CrewCls: Any) -> None:
    """Install the class-level WORKFLOW hooks on the Crew class.

    kickoff is the canonical root; kickoff_for_each(_async) open a batch root
    (their per-item copies re-enter the patched kickoff for the item spans);
    train/test/replay each open their own root. kickoff_async is intentionally
    NOT patched — it delegates to kickoff via asyncio.to_thread."""
    if getattr(CrewCls, "_neatlogs_class_patched", False):
        return

    # --- kickoff (canonical WORKFLOW root) ---
    if "kickoff" in CrewCls.__dict__:
        orig_kickoff = CrewCls.kickoff

        def patched_kickoff(self, *args, **kwargs):
            if _kickoff_in_progress(self):
                return orig_kickoff(self, *args, **kwargs)
            _dbg(
                f"kickoff() FIRED on crew id={hex(id(self))} "
                f"(name={getattr(self, 'name', None) or getattr(self, '_name', None)!r}) "
                f"— called from {_caller()}"
            )
            # Catch provider LLM / tool subclasses imported after class hooks ran.
            _patch_tasks_and_agents(self)
            tracer = get_tracer()
            span = tracer.start_span(
                name="crewai.crew.kickoff",
                attributes=_get_crew_attributes(self),
                **_shared_trace_kwargs(),
            )
            _dbg(f"opened 'crewai.crew.kickoff' WORKFLOW span for crew id={hex(id(self))}")
            _set_crew_input(span, self, kwargs)
            token = attach_as_current(span)
            _mark_kickoff(self, True)
            start = time.perf_counter()
            try:
                result = orig_kickoff(self, *args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                _mark_kickoff(self, False)
                detach(token)
            _finalize_crew_span(span, result, (time.perf_counter() - start) * 1000)
            return result

        CrewCls.kickoff = patched_kickoff

    # --- kickoff_for_each / kickoff_for_each_async (batch roots) ---
    if "kickoff_for_each" in CrewCls.__dict__:
        orig_kfe = CrewCls.kickoff_for_each

        def patched_kfe(self, *args, **kwargs):
            tracer = get_tracer()
            inputs = kwargs.get("inputs") or (args[0] if args else None)
            attrs = _get_crew_attributes(self)
            if inputs and hasattr(inputs, "__len__"):
                attrs["neatlogs.workflow.batch_size"] = len(inputs)
            if inputs:
                attrs["input.value"] = serialize(inputs)[:10000]
            span = tracer.start_span(name="crewai.crew.kickoff_for_each", attributes=attrs)
            token = attach_as_current(span)
            start = time.perf_counter()
            try:
                results = orig_kfe(self, *args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                detach(token)
            if results is not None:
                try:
                    span.set_attribute(
                        "output.value",
                        serialize([getattr(r, "raw", str(r)) for r in results])[:10000],
                    )
                except TypeError:
                    pass
            span.set_attribute(
                "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
            )
            span.set_status(StatusCode.OK)
            span.end()
            return results

        CrewCls.kickoff_for_each = patched_kfe

    if "kickoff_for_each_async" in CrewCls.__dict__:
        orig_kfea = CrewCls.kickoff_for_each_async

        async def patched_kfea(self, *args, **kwargs):
            tracer = get_tracer()
            inputs = kwargs.get("inputs") or (args[0] if args else None)
            attrs = _get_crew_attributes(self)
            if inputs and hasattr(inputs, "__len__"):
                attrs["neatlogs.workflow.batch_size"] = len(inputs)
            if inputs:
                attrs["input.value"] = serialize(inputs)[:10000]
            span = tracer.start_span(name="crewai.crew.kickoff_for_each_async", attributes=attrs)
            token = attach_as_current(span)
            start = time.perf_counter()
            try:
                results = await orig_kfea(self, *args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                detach(token)
            span.set_attribute(
                "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
            )
            span.set_status(StatusCode.OK)
            span.end()
            return results

        CrewCls.kickoff_for_each_async = patched_kfea

    # --- akickoff (native async in CrewAI 1.x; ABSENT in 0.130.0) + train/test/replay ---
    _patch_extra_crew_class_entrypoints(CrewCls)

    CrewCls._neatlogs_class_patched = True


# akickoff is a DISTINCT coroutine (does NOT delegate to kickoff) → needs its own
# span. train/test/replay are sync run entrypoints. All version-guarded: only
# patched if the class actually defines them.
_EXTRA_CREW_ENTRYPOINTS = {
    "akickoff": ("crewai.crew.akickoff", True),
    "akickoff_for_each": ("crewai.crew.akickoff_for_each", True),
    "train": ("crewai.crew.train", False),
    "test": ("crewai.crew.test", False),
    "replay": ("crewai.crew.replay", False),
}


def _patch_extra_crew_class_entrypoints(CrewCls: Any) -> None:
    for method_name, (span_name, is_async) in _EXTRA_CREW_ENTRYPOINTS.items():
        if method_name not in CrewCls.__dict__:
            continue
        orig = CrewCls.__dict__[method_name]

        def _make(orig=orig, span_name=span_name, is_async=is_async):
            def _open(self, kwargs):
                _patch_tasks_and_agents(self)
                span = get_tracer().start_span(
                    name=span_name, attributes=_get_crew_attributes(self)
                )
                _set_crew_input(span, self, kwargs)
                return span

            if is_async:

                async def patched(self, *args, **kwargs):
                    span = _open(self, kwargs)
                    token = attach_as_current(span)
                    start = time.perf_counter()
                    try:
                        result = await orig(self, *args, **kwargs)
                    except Exception as e:
                        _err(span, e)
                        raise
                    finally:
                        detach(token)
                    _finalize_crew_span(span, result, (time.perf_counter() - start) * 1000)
                    return result

                return patched

            def patched(self, *args, **kwargs):
                span = _open(self, kwargs)
                token = attach_as_current(span)
                start = time.perf_counter()
                try:
                    result = orig(self, *args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                finally:
                    detach(token)
                _finalize_crew_span(span, result, (time.perf_counter() - start) * 1000)
                return result

            return patched

        setattr(CrewCls, method_name, _make())


# ---------------------------------------------------------------------------
# Task (TASK spans) + Agent (AGENT spans)
# ---------------------------------------------------------------------------


def _patch_tasks_and_agents(crew: Any) -> None:
    for task in getattr(crew, "tasks", []) or []:
        _patch_task_execute(task)
    for agent in getattr(crew, "agents", []) or []:
        _patch_agent_execute(agent)
        # Patch this agent's concrete LLM class in case its provider subclass was
        # imported only after the global class hooks ran.
        llm = getattr(agent, "llm", None)
        if llm is not None:
            cls = type(llm)
            if "call" in cls.__dict__ and not cls.__dict__.get("_neatlogs_patched", False):
                _patch_llm_call(cls)
        # Patch the concrete class of each tool this agent holds (covers tool
        # subclasses imported after the global hooks ran).
        for tool in getattr(agent, "tools", None) or []:
            tcls = type(tool)
            if "run" in tcls.__dict__ and not tcls.__dict__.get("_neatlogs_patched", False):
                _patch_tool_run(tcls)


def _patch_task_execute(task: Any) -> None:
    if getattr(task, "_neatlogs_task_patched", False):
        return

    def _attrs():
        attrs = _crewai_span_attributes("task")
        task_id = getattr(task, "id", None)
        if task_id:
            attrs["neatlogs.task.id"] = str(task_id)
        task_key = getattr(task, "key", None)
        if task_key:
            attrs["neatlogs.task.key"] = str(task_key)
        description = getattr(task, "description", "")
        if description:
            attrs["input.value"] = str(description)[:10000]
        agent = getattr(task, "agent", None)
        if agent:
            role = getattr(agent, "role", "")
            if role:
                attrs["neatlogs.agent.role"] = role
        return attrs

    def _finalize(span, result, start):
        if result is not None:
            raw = getattr(result, "raw", None) if hasattr(result, "raw") else str(result)
            if raw:
                span.set_attribute("output.value", str(raw)[:10000])
        span.set_attribute(
            "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
        )
        span.set_status(StatusCode.OK)
        span.end()

    # Sync core (covers execute_sync path)
    sync_method = (
        "_execute_core"
        if hasattr(task, "_execute_core")
        else ("execute_sync" if hasattr(task, "execute_sync") else None)
    )
    if sync_method:
        orig_sync = getattr(task, sync_method)

        def patched_sync(*args, **kwargs):
            tracer = get_tracer()
            span = tracer.start_span(name="crewai.task", attributes=_attrs())
            token = attach_as_current(span)
            start = time.perf_counter()
            try:
                result = orig_sync(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                detach(token)
            _finalize(span, result, start)
            return result

        _safe_setattr(task, sync_method, patched_sync)

    # Async execution
    if hasattr(task, "execute_async"):
        orig_async = task.execute_async

        def patched_async(*args, **kwargs):
            # execute_async returns a Future; wrap to time the whole task.
            tracer = get_tracer()
            span = tracer.start_span(name="crewai.task.async", attributes=_attrs())
            token = attach_as_current(span)
            start = time.perf_counter()
            try:
                future = orig_async(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                detach(token)
                raise
            detach(token)

            # Attach a done-callback to finalize when the future completes.
            def _done(fut):
                try:
                    result = fut.result()
                    _finalize(span, result, start)
                except Exception as e:
                    _err(span, e)

            try:
                future.add_done_callback(_done)
            except Exception:
                _finalize(span, None, start)
            return future

        _safe_setattr(task, "execute_async", patched_async)

    _safe_setattr(task, "_neatlogs_task_patched", True)


def _patch_agent_execute(agent: Any) -> None:
    if getattr(agent, "_neatlogs_agent_patched", False) or not hasattr(agent, "execute_task"):
        return
    orig = agent.execute_task

    def patched_execute_task(*args, **kwargs):
        tracer = get_tracer()
        attrs = _crewai_span_attributes("agent")
        role = getattr(agent, "role", "")
        if role:
            attrs["neatlogs.agent.role"] = role
        agent_name = getattr(agent, "name", None)
        if agent_name:
            attrs["neatlogs.agent.name"] = agent_name
        tools = getattr(agent, "tools", None)
        if tools:
            for i, tool in enumerate(tools):
                tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
                if tool_name:
                    attrs[f"neatlogs.llm.tools.{i}.name"] = str(tool_name)
                tool_desc = getattr(tool, "description", None)
                if tool_desc:
                    attrs[f"neatlogs.llm.tools.{i}.description"] = str(tool_desc)[:500]

        span = tracer.start_span(
            name=f"crewai.agent.{role}" if role else "crewai.agent", attributes=attrs
        )
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("output.value", str(result)[:10000])
        span.set_attribute(
            "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
        )
        span.set_status(StatusCode.OK)
        span.end()
        return result

    _safe_setattr(agent, "execute_task", patched_execute_task)
    _safe_setattr(agent, "_neatlogs_agent_patched", True)


def _agent_span_attributes(agent: Any) -> dict:
    attrs = _crewai_span_attributes("agent")
    role = getattr(agent, "role", "")
    if role:
        attrs["neatlogs.agent.role"] = role
    agent_name = getattr(agent, "name", None)
    if agent_name:
        attrs["neatlogs.agent.name"] = str(agent_name)
    return attrs


def _set_agent_messages_input(span: Any, kwargs: dict, args: tuple) -> None:
    """Capture Agent.kickoff input. `messages` is a string query or a list of
    {role, content} dicts; may be passed positionally or by keyword."""
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]
    if messages:
        span.set_attribute("input.value", serialize(messages)[:10000])


def _patch_agent_kickoff(agent: Any) -> None:
    """Standalone Agent execution (no Crew): agent.kickoff(messages=...).

    Emits an AGENT-kind root span (not a crew span). Tool/LLM class hooks are
    already installed by wrap(), so tool and model calls nest underneath.
    """
    role = getattr(agent, "role", "")
    span_name = f"crewai.agent.{role}" if role else "crewai.agent"

    def _open():
        span = get_tracer().start_span(name=span_name, attributes=_agent_span_attributes(agent))
        return span

    def _finish(span, result, start):
        if result is not None:
            raw = getattr(result, "raw", None)
            span.set_attribute("output.value", str(raw if raw is not None else result)[:10000])
        span.set_attribute(
            "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
        )
        span.set_status(StatusCode.OK)
        span.end()

    if hasattr(agent, "kickoff") and not getattr(agent, "_neatlogs_agent_kickoff_patched", False):
        orig = agent.kickoff

        def patched_kickoff(*args, **kwargs):
            span = _open()
            _set_agent_messages_input(span, kwargs, args)
            token = attach_as_current(span)
            start = time.perf_counter()
            try:
                result = orig(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                detach(token)
            _finish(span, result, start)
            return result

        _safe_setattr(agent, "kickoff", patched_kickoff)
        _safe_setattr(agent, "_neatlogs_agent_kickoff_patched", True)

    for method_name in ("kickoff_async", "akickoff"):
        if not hasattr(agent, method_name):
            continue
        flag = f"_neatlogs_agent_{method_name}_patched"
        if getattr(agent, flag, False):
            continue
        orig_async = getattr(agent, method_name)

        def _make(orig_async=orig_async):
            async def patched_async(*args, **kwargs):
                span = _open()
                _set_agent_messages_input(span, kwargs, args)
                token = attach_as_current(span)
                start = time.perf_counter()
                try:
                    result = await orig_async(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                finally:
                    detach(token)
                _finish(span, result, start)
                return result

            return patched_async

        _safe_setattr(agent, method_name, _make())
        _safe_setattr(agent, flag, True)


# ---------------------------------------------------------------------------
# Flow (WORKFLOW spans)
# ---------------------------------------------------------------------------


def _set_flow_input(span: Any, flow: Any, kwargs: dict) -> None:
    """Set input.value on a flow span.

    Prefer the explicit ``inputs=`` kwarg. When absent, fall back to the flow's
    ``state`` (Pydantic model or dict) captured at span open — the closest thing
    a Flow has to an input, so the workflow root isn't left empty.
    """
    inputs = kwargs.get("inputs")
    if inputs:
        span.set_attribute("input.value", serialize(inputs))
        return
    state = getattr(flow, "state", None)
    if state is not None:
        try:
            span.set_attribute("input.value", serialize(state)[:10000])
        except Exception:
            pass


def _patch_flow_class(FlowCls: Any) -> None:
    """Patch a Flow subclass at the class level.

    CrewAI's ``Flow.kickoff`` delegates to ``kickoff_async`` (``asyncio.run``), so
    patching both would double-count. The re-entrancy guard opens exactly one
    WORKFLOW span for the outermost of the two, whichever the user calls."""
    if getattr(FlowCls, "_neatlogs_flow_patched", False):
        return

    def _attrs(flow):
        attrs = _crewai_span_attributes("workflow")
        attrs.update(
            {
                "neatlogs.workflow.type": "flow",
                "neatlogs.workflow.name": type(flow).__name__,
            }
        )
        return attrs

    if "kickoff" in FlowCls.__dict__:
        orig = FlowCls.kickoff

        def patched_kickoff(self, *args, **kwargs):
            if _kickoff_in_progress(self):
                return orig(self, *args, **kwargs)
            tracer = get_tracer()
            span = tracer.start_span(name="crewai.flow.kickoff", attributes=_attrs(self))
            _set_flow_input(span, self, kwargs)
            token = attach_as_current(span)
            _mark_kickoff(self, True)
            start = time.perf_counter()
            try:
                result = orig(self, *args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                _mark_kickoff(self, False)
                detach(token)
            if result is not None:
                span.set_attribute("output.value", str(result)[:10000])
            span.set_attribute(
                "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
            )
            span.set_status(StatusCode.OK)
            span.end()
            return result

        FlowCls.kickoff = patched_kickoff

    if "kickoff_async" in FlowCls.__dict__:
        orig_async = FlowCls.kickoff_async

        async def patched_kickoff_async(self, *args, **kwargs):
            if _kickoff_in_progress(self):
                return await orig_async(self, *args, **kwargs)
            tracer = get_tracer()
            span = tracer.start_span(name="crewai.flow.kickoff_async", attributes=_attrs(self))
            _set_flow_input(span, self, kwargs)
            token = attach_as_current(span)
            _mark_kickoff(self, True)
            start = time.perf_counter()
            try:
                result = await orig_async(self, *args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            finally:
                _mark_kickoff(self, False)
                detach(token)
            if result is not None:
                span.set_attribute("output.value", str(result)[:10000])
            span.set_attribute(
                "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
            )
            span.set_status(StatusCode.OK)
            span.end()
            return result

        FlowCls.kickoff_async = patched_kickoff_async

    FlowCls._neatlogs_flow_patched = True


# ---------------------------------------------------------------------------
# Class-level hooks: TOOL (BaseTool.run) + LLM (LLM.call)
# ---------------------------------------------------------------------------


def _install_class_hooks() -> None:
    global _CLASS_HOOKS_INSTALLED
    if _CLASS_HOOKS_INSTALLED:
        return
    _CLASS_HOOKS_INSTALLED = True
    _patch_crew_classes()
    _patch_base_tool()
    _patch_structured_tool()
    _patch_llm()


def _patch_crew_classes() -> None:
    """Patch the Crew class (WORKFLOW roots). Task/Agent spans are opened per
    instance from inside the class-level kickoff via ``_patch_tasks_and_agents``,
    so they're covered for wrapped and bare crews alike."""
    try:
        from crewai.crew import Crew

        _patch_crew_class(Crew)
    except Exception as e:
        _dbg(f"could not class-patch Crew: {e}")


def _patch_base_tool() -> None:
    try:
        from crewai.tools.base_tool import BaseTool
    except Exception:
        return
    # BaseTool subclasses (Tool, CrewStructuredTool, agent tools, custom tools)
    # each override run(); patch the base AND every subclass that defines its own
    # run, so the @tool-decorated objects (class Tool) are covered.
    targets = [BaseTool] + _all_subclasses(BaseTool)
    for cls in targets:
        if "run" in cls.__dict__ and not cls.__dict__.get("_neatlogs_patched", False):
            _patch_tool_run(cls)


def _patch_tool_run(ToolCls) -> None:
    orig_run = ToolCls.run

    def patched_run(self, *args, **kwargs):
        tracer = get_tracer()
        attrs = _crewai_span_attributes("tool")
        name = getattr(self, "name", None) or type(self).__name__
        attrs["neatlogs.tool.name"] = str(name)
        desc = getattr(self, "description", None)
        if desc:
            attrs["neatlogs.tool.description"] = str(desc)[:500]
        if kwargs:
            attrs["input.value"] = _serialize_crewai_io(kwargs)
        elif args:
            attrs["input.value"] = _serialize_crewai_io(args)

        span = tracer.start_span(name=f"crewai.tool.{name}", attributes=attrs)
        token = attach_as_current(span)
        try:
            result = orig_run(self, *args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("output.value", _serialize_crewai_io(result))
        span.set_status(StatusCode.OK)
        span.end()
        return result

    ToolCls.run = patched_run
    ToolCls._neatlogs_patched = True


def _patch_structured_tool() -> None:
    """CrewStructuredTool is used for function tools and invokes via _run/invoke."""
    try:
        from crewai.tools.structured_tool import CrewStructuredTool
    except Exception:
        return
    if getattr(CrewStructuredTool, "_neatlogs_patched", False):
        return

    target_method = (
        "invoke"
        if "invoke" in CrewStructuredTool.__dict__
        else ("_run" if "_run" in CrewStructuredTool.__dict__ else None)
    )
    if not target_method:
        return
    orig = getattr(CrewStructuredTool, target_method)

    def patched(self, *args, **kwargs):
        tracer = get_tracer()
        name = getattr(self, "name", None) or type(self).__name__
        attrs = _crewai_span_attributes("tool")
        attrs["neatlogs.tool.name"] = str(name)
        desc = getattr(self, "description", None)
        if desc:
            attrs["neatlogs.tool.description"] = str(desc)[:500]
        payload = kwargs if kwargs else (args[0] if args else None)
        if payload is not None:
            attrs["input.value"] = _serialize_crewai_io(payload)
        span = tracer.start_span(name=f"crewai.tool.{name}", attributes=attrs)
        token = attach_as_current(span)
        try:
            result = orig(self, *args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("output.value", _serialize_crewai_io(result))
        span.set_status(StatusCode.OK)
        span.end()
        return result

    setattr(CrewStructuredTool, target_method, patched)
    CrewStructuredTool._neatlogs_patched = True


def _patch_llm() -> None:
    # CrewAI routes LLMs through BaseLLM subclasses (native providers like
    # crewai.llms.providers.openai.completion.OpenAICompletion override .call),
    # plus the legacy crewai.llm.LLM. Patch the base and every concrete subclass
    # that defines its own call().
    targets = []
    try:
        from crewai.llms.base_llm import BaseLLM

        targets.append(BaseLLM)
        targets.extend(_all_subclasses(BaseLLM))
    except Exception:
        pass
    try:
        from crewai.llm import LLM

        if LLM not in targets:
            targets.append(LLM)
    except Exception:
        pass

    for cls in targets:
        if "call" in cls.__dict__ and not cls.__dict__.get("_neatlogs_patched", False):
            _patch_llm_call(cls)


def _all_subclasses(cls):
    seen = set()
    stack = list(cls.__subclasses__())
    out = []
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        stack.extend(c.__subclasses__())
    return out


# CrewAI's LLM.call goes through litellm.completion(**params) and returns ONLY a
# string — per-call usage/model/provider/request-params never surface on the LLM
# object (there is no get_token_usage_summary on current versions). We wrap
# litellm.completion ONCE to record the last completion's usage + request params
# into a thread-local, then read it back in patched_call after the call returns.
# This records data only (no spans), so isolation is untouched.
_litellm_capture = threading.local()
_LITELLM_WRAPPED = False


def _install_litellm_capture() -> None:
    global _LITELLM_WRAPPED
    if _LITELLM_WRAPPED:
        return
    try:
        import litellm
    except Exception:
        return

    def _record(resp: Any, kwargs: dict) -> None:
        try:
            rec: dict = {}
            usage = getattr(resp, "usage", None)
            if usage is not None:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = getattr(usage, k, None)
                    if v is None and isinstance(usage, dict):
                        v = usage.get(k)
                    if isinstance(v, (int, float)):
                        rec[k] = v
                ptd = getattr(usage, "prompt_tokens_details", None)
                if ptd is not None:
                    cached = getattr(ptd, "cached_tokens", None)
                    if isinstance(cached, (int, float)):
                        rec["cached_tokens"] = cached
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is not None:
                    reasoning = getattr(ctd, "reasoning_tokens", None)
                    if isinstance(reasoning, (int, float)):
                        rec["reasoning_tokens"] = reasoning
            model = getattr(resp, "model", None)
            if model:
                rec["model"] = str(model)
            hp = getattr(resp, "_hidden_params", None) or {}
            if isinstance(hp, dict):
                if hp.get("custom_llm_provider"):
                    rec["provider"] = str(hp["custom_llm_provider"])
            # Request params AS SENT (crewai maps max_completion_tokens→max_tokens here).
            params = {}
            for p in (
                "temperature",
                "top_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            ):
                v = kwargs.get(p)
                if v is not None:
                    params[p] = v
            if params:
                rec["params"] = params
            _litellm_capture.last = rec
        except Exception:
            pass

    orig_completion = litellm.completion

    def _wrapped_completion(*args, **kwargs):
        resp = orig_completion(*args, **kwargs)
        # Non-streaming ModelResponse has .usage immediately; streams don't — skip
        # those (crewai's streaming path tracks usage on its own).
        if getattr(resp, "usage", None) is not None:
            _record(resp, kwargs)
        return resp

    litellm.completion = _wrapped_completion
    _LITELLM_WRAPPED = True


def _patch_llm_call(LLM) -> None:
    _install_litellm_capture()
    orig_call = LLM.call

    def patched_call(self, messages, *args, **kwargs):
        tracer = get_tracer()
        attrs = _crewai_span_attributes("llm")
        attrs["neatlogs.llm.provider"] = "crewai"
        model = getattr(self, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)

        input_messages: list[dict[str, str]] = []
        if isinstance(messages, str):
            input_messages.append(
                {
                    "role": "user",
                    "content": messages[:_LLM_INPUT_CAPTURE_LIMIT],
                }
            )
        elif isinstance(messages, list):
            for msg in messages:
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                content = (
                    msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                )
                content_text = content if isinstance(content, str) else serialize(content)
                input_messages.append(
                    {
                        "role": str(role) if role else "",
                        "content": content_text[:_LLM_INPUT_CAPTURE_LIMIT],
                    }
                )

        for i, message in enumerate(input_messages):
            if message["role"]:
                attrs[f"neatlogs.llm.input_messages.{i}.role"] = message["role"]
            if message["content"]:
                attrs[f"neatlogs.llm.input_messages.{i}.content"] = message["content"]
        if input_messages:
            attrs["input.value"] = serialize({"messages": input_messages})

        # Invocation params from the LLM object (max_completion_tokens is crewai's
        # alias for max_tokens). The litellm capture below overrides these with the
        # params AS SENT when available — this is the fallback for streaming/mocks.
        req_params = {}
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_tokens", "max_tokens"),
            ("max_completion_tokens", "max_tokens"),
            ("frequency_penalty", "frequency_penalty"),
            ("presence_penalty", "presence_penalty"),
        ):
            v = getattr(self, src, None)
            if v is not None and dst not in req_params:
                req_params[dst] = v

        # Clear any stale capture so we only read THIS call's record.
        _litellm_capture.last = None
        span = tracer.start_span(name="crewai.llm.call", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig_call(self, messages, *args, **kwargs)
        except Exception as e:
            _err(span, e)
            raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
            span.set_attribute("neatlogs.llm.output_messages.0.content", str(result)[:10000])

        # Read what our litellm.completion wrapper recorded for THIS call: real
        # per-call tokens, resolved provider (e.g. "azure"), response model, and the
        # request params as litellm received them.
        rec = getattr(_litellm_capture, "last", None) or {}
        if rec.get("params"):
            req_params.update(rec["params"])
        if rec.get("provider"):
            attrs["neatlogs.llm.provider"] = rec["provider"]
            span.set_attribute("neatlogs.llm.provider", rec["provider"])
        if rec.get("model"):
            span.set_attribute("neatlogs.llm.model_name", rec["model"])

        # Emit invocation_parameters as the JSON blob the backend reads for
        # model_settings (flat keys are NOT read there); keep flat keys too for
        # any consumer that wants them.
        if req_params:
            span.set_attribute("neatlogs.llm.invocation_parameters", json.dumps(req_params))
            for k, v in req_params.items():
                span.set_attribute(f"neatlogs.llm.{k}", v)

        for dst, key in (
            ("prompt", "prompt_tokens"),
            ("completion", "completion_tokens"),
            ("total", "total_tokens"),
            ("cache_read", "cached_tokens"),
            ("reasoning", "reasoning_tokens"),
        ):
            v = rec.get(key)
            if isinstance(v, (int, float)) and v > 0:
                span.set_attribute(f"neatlogs.llm.token_count.{dst}", v)

        span.set_attribute(
            "neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3)
        )
        span.set_status(StatusCode.OK)
        span.end()
        return result

    LLM.call = patched_call
    LLM._neatlogs_patched = True


def _err(span: Any, e: Exception) -> None:  # noqa: E305
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()
