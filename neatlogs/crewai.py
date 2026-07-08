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

Crew/task/agent spans are patched on the instance; TOOL and LLM spans are
installed once at the class level so every tool call and model call nests under
the active task/agent — including tools/agents/tasks added after wrap().
"""

import os
import time
from typing import Any

from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize

_CLASS_HOOKS_INSTALLED = False


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


def wrap_crewai(obj: Any) -> Any:
    """
    Wrap a CrewAI Crew or Flow instance.
    Returns the same instance with full span-hierarchy tracing.
    """
    _suppress_crewai_telemetry()
    _install_class_hooks()

    cls_name = type(obj).__name__
    module = type(obj).__module__ or ""

    # Flow detection
    if "flow" in module or hasattr(obj, "_methods") and hasattr(obj, "kickoff") and not hasattr(obj, "tasks"):
        _patch_flow(obj)
        return obj

    # Standalone Agent (no Crew): agent.kickoff(messages=...). Has a `role` and
    # `execute_task` but no `tasks`/`agents` — must NOT go through the Crew branch,
    # which would mislabel the run as `crewai.crew.kickoff`.
    if (
        hasattr(obj, "kickoff")
        and hasattr(obj, "execute_task")
        and hasattr(obj, "role")
        and not hasattr(obj, "tasks")
        and not hasattr(obj, "agents")
    ):
        _patch_agent_kickoff(obj)
        return obj

    # Crew
    _patch_kickoff(obj)
    _patch_kickoff_async(obj)
    _patch_kickoff_for_each(obj)
    _patch_kickoff_for_each_async(obj)
    _patch_extra_crew_entrypoints(obj)
    _patch_tasks_and_agents(obj)
    return obj


# ---------------------------------------------------------------------------
# Crew (WORKFLOW spans)
# ---------------------------------------------------------------------------


def _get_crew_attributes(crew: Any) -> dict:
    attrs = {"neatlogs.span.kind": "workflow"}

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
        attrs["neatlogs.crewai.process"] = str(process.value) if hasattr(process, "value") else str(process)

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
    usage = token_usage if isinstance(token_usage, dict) else (token_usage.__dict__ if hasattr(token_usage, "__dict__") else {})
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


def _patch_kickoff(crew: Any) -> None:
    if getattr(crew, "_neatlogs_kickoff_patched", False):
        return
    orig_kickoff = crew.kickoff

    def patched_kickoff(*args, **kwargs):
        # Re-patch in case tasks/agents were added after wrap().
        _patch_tasks_and_agents(crew)
        tracer = get_tracer()
        attrs = _get_crew_attributes(crew)
        span = tracer.start_span(name="crewai.crew.kickoff", attributes=attrs)
        _set_crew_input(span, crew, kwargs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig_kickoff(*args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        _finalize_crew_span(span, result, (time.perf_counter() - start) * 1000)
        return result

    _safe_setattr(crew, "kickoff", patched_kickoff)
    _safe_setattr(crew, "_neatlogs_kickoff_patched", True)


def _patch_kickoff_async(crew: Any) -> None:
    if not hasattr(crew, "kickoff_async") or getattr(crew, "_neatlogs_kickoff_async_patched", False):
        return
    orig = crew.kickoff_async

    async def patched(*args, **kwargs):
        _patch_tasks_and_agents(crew)
        tracer = get_tracer()
        attrs = _get_crew_attributes(crew)
        span = tracer.start_span(name="crewai.crew.kickoff_async", attributes=attrs)
        _set_crew_input(span, crew, kwargs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = await orig(*args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        _finalize_crew_span(span, result, (time.perf_counter() - start) * 1000)
        return result

    _safe_setattr(crew, "kickoff_async", patched)
    _safe_setattr(crew, "_neatlogs_kickoff_async_patched", True)


def _patch_kickoff_for_each(crew: Any) -> None:
    if not hasattr(crew, "kickoff_for_each") or getattr(crew, "_neatlogs_kfe_patched", False):
        return
    orig = crew.kickoff_for_each

    def patched(*args, **kwargs):
        _patch_tasks_and_agents(crew)
        tracer = get_tracer()
        inputs = kwargs.get("inputs") or (args[0] if args else None)
        attrs = _get_crew_attributes(crew)
        if inputs and hasattr(inputs, "__len__"):
            attrs["neatlogs.workflow.batch_size"] = len(inputs)
        if inputs:
            attrs["input.value"] = serialize(inputs)[:10000]
        span = tracer.start_span(name="crewai.crew.kickoff_for_each", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            results = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        if results is not None:
            try:
                span.set_attribute("output.value", serialize([getattr(r, "raw", str(r)) for r in results])[:10000])
            except TypeError:
                pass
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return results

    _safe_setattr(crew, "kickoff_for_each", patched)
    _safe_setattr(crew, "_neatlogs_kfe_patched", True)


def _patch_kickoff_for_each_async(crew: Any) -> None:
    if not hasattr(crew, "kickoff_for_each_async") or getattr(crew, "_neatlogs_kfea_patched", False):
        return
    orig = crew.kickoff_for_each_async

    async def patched(*args, **kwargs):
        _patch_tasks_and_agents(crew)
        tracer = get_tracer()
        inputs = kwargs.get("inputs") or (args[0] if args else None)
        attrs = _get_crew_attributes(crew)
        if inputs and hasattr(inputs, "__len__"):
            attrs["neatlogs.workflow.batch_size"] = len(inputs)
        if inputs:
            attrs["input.value"] = serialize(inputs)[:10000]
        span = tracer.start_span(name="crewai.crew.kickoff_for_each_async", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            results = await orig(*args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return results

    _safe_setattr(crew, "kickoff_for_each_async", patched)
    _safe_setattr(crew, "_neatlogs_kfea_patched", True)


# Additional Crew entrypoints: async aliases (akickoff*) + train / test / replay.
# CrewAI exposes akickoff as a DISTINCT coroutine (not kickoff_async), and
# train/test/replay are sync run entrypoints (declared as project scripts) that
# would otherwise emit no span at all.

_EXTRA_CREW_ENTRYPOINTS = {
    "akickoff": ("crewai.crew.akickoff", True),
    "akickoff_for_each": ("crewai.crew.akickoff_for_each", True),
    "train": ("crewai.crew.train", False),
    "test": ("crewai.crew.test", False),
    "replay": ("crewai.crew.replay", False),
}


def _patch_extra_crew_entrypoints(crew: Any) -> None:
    for method_name, (span_name, is_async) in _EXTRA_CREW_ENTRYPOINTS.items():
        if not hasattr(crew, method_name):
            continue
        flag = f"_neatlogs_{method_name}_patched"
        if getattr(crew, flag, False):
            continue
        orig = getattr(crew, method_name)

        def _make(orig=orig, span_name=span_name, is_async=is_async):
            def _open(kwargs):
                _patch_tasks_and_agents(crew)
                span = get_tracer().start_span(
                    name=span_name, attributes=_get_crew_attributes(crew)
                )
                _set_crew_input(span, crew, kwargs)
                return span

            if is_async:
                async def patched(*args, **kwargs):
                    span = _open(kwargs)
                    token = attach_as_current(span)
                    start = time.perf_counter()
                    try:
                        result = await orig(*args, **kwargs)
                    except Exception as e:
                        _err(span, e); raise
                    finally:
                        detach(token)
                    _finalize_crew_span(span, result, (time.perf_counter() - start) * 1000)
                    return result
                return patched

            def patched(*args, **kwargs):
                span = _open(kwargs)
                token = attach_as_current(span)
                start = time.perf_counter()
                try:
                    result = orig(*args, **kwargs)
                except Exception as e:
                    _err(span, e); raise
                finally:
                    detach(token)
                _finalize_crew_span(span, result, (time.perf_counter() - start) * 1000)
                return result
            return patched

        _safe_setattr(crew, method_name, _make())
        _safe_setattr(crew, flag, True)


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
        attrs = {"neatlogs.span.kind": "task"}
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
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()

    # Sync core (covers execute_sync path)
    sync_method = "_execute_core" if hasattr(task, "_execute_core") else ("execute_sync" if hasattr(task, "execute_sync") else None)
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
                _err(span, e); raise
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
                _err(span, e); detach(token); raise
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
        attrs = {"neatlogs.span.kind": "agent"}
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

        span = tracer.start_span(name=f"crewai.agent.{role}" if role else "crewai.agent", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig(*args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("output.value", str(result)[:10000])
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return result

    _safe_setattr(agent, "execute_task", patched_execute_task)
    _safe_setattr(agent, "_neatlogs_agent_patched", True)


def _agent_span_attributes(agent: Any) -> dict:
    attrs = {"neatlogs.span.kind": "agent"}
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
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
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
                _err(span, e); raise
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
                    _err(span, e); raise
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


def _patch_flow(flow: Any) -> None:
    if getattr(flow, "_neatlogs_flow_patched", False):
        return

    def _attrs():
        attrs = {"neatlogs.span.kind": "workflow", "neatlogs.workflow.type": "flow"}
        name = type(flow).__name__
        attrs["neatlogs.workflow.name"] = name
        return attrs

    if hasattr(flow, "kickoff"):
        orig = flow.kickoff

        def patched_kickoff(*args, **kwargs):
            tracer = get_tracer()
            attrs = _attrs()
            span = tracer.start_span(name="crewai.flow.kickoff", attributes=attrs)
            _set_flow_input(span, flow, kwargs)
            token = attach_as_current(span)
            start = time.perf_counter()
            try:
                result = orig(*args, **kwargs)
            except Exception as e:
                _err(span, e); raise
            finally:
                detach(token)
            if result is not None:
                span.set_attribute("output.value", str(result)[:10000])
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
            span.set_status(StatusCode.OK)
            span.end()
            return result

        _safe_setattr(flow, "kickoff", patched_kickoff)

    if hasattr(flow, "kickoff_async"):
        orig_async = flow.kickoff_async

        async def patched_kickoff_async(*args, **kwargs):
            tracer = get_tracer()
            attrs = _attrs()
            span = tracer.start_span(name="crewai.flow.kickoff_async", attributes=attrs)
            _set_flow_input(span, flow, kwargs)
            token = attach_as_current(span)
            start = time.perf_counter()
            try:
                result = await orig_async(*args, **kwargs)
            except Exception as e:
                _err(span, e); raise
            finally:
                detach(token)
            if result is not None:
                span.set_attribute("output.value", str(result)[:10000])
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
            span.set_status(StatusCode.OK)
            span.end()
            return result

        _safe_setattr(flow, "kickoff_async", patched_kickoff_async)

    _safe_setattr(flow, "_neatlogs_flow_patched", True)


# ---------------------------------------------------------------------------
# Class-level hooks: TOOL (BaseTool.run) + LLM (LLM.call)
# ---------------------------------------------------------------------------


def _install_class_hooks() -> None:
    global _CLASS_HOOKS_INSTALLED
    if _CLASS_HOOKS_INSTALLED:
        return
    _CLASS_HOOKS_INSTALLED = True
    _patch_base_tool()
    _patch_structured_tool()
    _patch_llm()


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
        attrs = {"neatlogs.span.kind": "tool"}
        name = getattr(self, "name", None) or type(self).__name__
        attrs["neatlogs.tool.name"] = str(name)
        desc = getattr(self, "description", None)
        if desc:
            attrs["neatlogs.tool.description"] = str(desc)[:500]
        if kwargs:
            attrs["input.value"] = serialize(kwargs)[:10000]
        elif args:
            attrs["input.value"] = serialize(args)[:10000]

        span = tracer.start_span(name=f"crewai.tool.{name}", attributes=attrs)
        token = attach_as_current(span)
        try:
            result = orig_run(self, *args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("output.value", str(result)[:10000])
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

    target_method = "invoke" if "invoke" in CrewStructuredTool.__dict__ else ("_run" if "_run" in CrewStructuredTool.__dict__ else None)
    if not target_method:
        return
    orig = getattr(CrewStructuredTool, target_method)

    def patched(self, *args, **kwargs):
        tracer = get_tracer()
        name = getattr(self, "name", None) or type(self).__name__
        attrs = {"neatlogs.span.kind": "tool", "neatlogs.tool.name": str(name)}
        desc = getattr(self, "description", None)
        if desc:
            attrs["neatlogs.tool.description"] = str(desc)[:500]
        payload = kwargs if kwargs else (args[0] if args else None)
        if payload is not None:
            attrs["input.value"] = serialize(payload)[:10000]
        span = tracer.start_span(name=f"crewai.tool.{name}", attributes=attrs)
        token = attach_as_current(span)
        try:
            result = orig(self, *args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("output.value", str(result)[:10000])
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


def _crewai_usage_snapshot(llm: Any) -> dict:
    """Cumulative token usage from a crewai LLM via get_token_usage_summary()."""
    try:
        summary = llm.get_token_usage_summary()
    except Exception:
        return {}
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "cached_prompt_tokens", "reasoning_tokens"):
        v = getattr(summary, key, None)
        if v is None and isinstance(summary, dict):
            v = summary.get(key)
        if isinstance(v, (int, float)):
            out[key] = v
    return out


def _patch_llm_call(LLM) -> None:
    orig_call = LLM.call

    def patched_call(self, messages, *args, **kwargs):
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": "crewai"}
        model = getattr(self, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)

        if isinstance(messages, str):
            attrs["neatlogs.llm.input_messages.0.role"] = "user"
            attrs["neatlogs.llm.input_messages.0.content"] = messages[:10000]
        elif isinstance(messages, list):
            for i, msg in enumerate(messages):
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if role:
                    attrs[f"neatlogs.llm.input_messages.{i}.role"] = role
                if content:
                    attrs[f"neatlogs.llm.input_messages.{i}.content"] = (content if isinstance(content, str) else serialize(content))[:10000]

        for p in ("temperature", "max_tokens", "top_p"):
            v = getattr(self, p, None)
            if v is not None:
                attrs[f"neatlogs.llm.{p}"] = v

        # crewai's LLM.call returns only a string (no per-call usage). The LLM
        # object tracks CUMULATIVE usage via get_token_usage_summary(); snapshot it
        # before/after and emit the DELTA so each LLM span carries its own tokens
        # (otherwise tokens only exist as an aggregate on the kickoff span, which the
        # backend doesn't surface for non-LLM kinds).
        before = _crewai_usage_snapshot(self)

        span = tracer.start_span(name="crewai.llm.call", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig_call(self, messages, *args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)
        if result is not None:
            span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
            span.set_attribute("neatlogs.llm.output_messages.0.content", str(result)[:10000])
        after = _crewai_usage_snapshot(self)
        for dst, key in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"),
                         ("total", "total_tokens"), ("cache_read", "cached_prompt_tokens"),
                         ("reasoning", "reasoning_tokens")):
            delta = after.get(key, 0) - before.get(key, 0)
            if delta > 0:
                span.set_attribute(f"neatlogs.llm.token_count.{dst}", delta)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return result

    LLM.call = patched_call
    LLM._neatlogs_patched = True


def _err(span: Any, e: Exception) -> None:  # noqa: E305
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()
