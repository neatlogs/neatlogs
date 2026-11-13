"""Opt-in interception of decorated JSON tools, including async factories."""

from __future__ import annotations

import functools
import inspect

from .contracts import ReplayError
from .session import ReplayHit, current_replay, lookup_replay


def replay_tool(func, *, name, result):
    if result is not True and result != "awaitable":
        raise ReplayError("Tool replay accepts True or 'awaitable'.")
    if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
        raise ReplayError("Tool replay requires a complete JSON result, not a generator.")
    signature = inspect.signature(func)

    def decision(args, kwargs):
        if current_replay() is None:
            return None
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        for key, parameter in signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                values[key] = list(values[key])
        return lookup_replay("tool", name, values)

    if result == "awaitable" and not inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        def awaitable_wrapper(*args, **kwargs):
            replay = decision(args, kwargs)
            if isinstance(replay, ReplayHit):

                async def saved():
                    return replay.entry.output

                return saved()
            value = func(*args, **kwargs)
            if not inspect.isawaitable(value):
                raise ReplayError("This replay tool must return an awaitable.")
            return value

        return awaitable_wrapper

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            replay = decision(args, kwargs)
            if isinstance(replay, ReplayHit):
                return replay.entry.output
            value = func(*args, **kwargs)
            if not inspect.isawaitable(value):
                raise ReplayError("This replay tool must return an awaitable.")
            return await value

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        replay = decision(args, kwargs)
        if isinstance(replay, ReplayHit):
            return replay.entry.output
        return func(*args, **kwargs)

    return sync_wrapper
