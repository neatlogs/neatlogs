# Session agent connection (v2)

Import `AgentClient` from `neatlogs.agent`. This adapter uses the server's persistent enrollment for **session** units; it does not support end-user assignment across conversations or make rollout decisions. Existing `neatlogs.ab_test` remains protocol v1.

```python
from neatlogs.agent import AgentClient
import neatlogs
import asyncio

agents = AgentClient(base_url=base_url, api_key=api_key, keys=["support"])
# Inside your async handler, before opening the root trace:
version = await agents.choose_agent("support", unit=conversation_id,
    attributes={"environment": "production"}, fallback={"model": deployed_model})
trace_id = None
try:
    with neatlogs.trace("reply", session_id=conversation_id) as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        if version.reason == "assigned" and not agents.record_agent_run(version):
            raise RuntimeError("Unable to record assigned execution")
        await run_agent(version.params)
finally:
    await asyncio.to_thread(neatlogs.flush)
    if conversation_finished and trace_id:
        agents.complete_agent_decision(version, trace_id)
    await agents.flush_async()
# At process shutdown, after flushing:
agents.close()
```

Reuse one client. `choose_agent` performs bounded HTTP when config or the session's lease needs renewal; it runs blocking transport in a worker thread. `refresh_async()` explicitly refreshes config. Optional synchronous `start()` uses the existing daemon refresh/retry loop; do not call a blocking startup method on an event loop. Use the yielded Neatlogs span for trace identity, since the global OpenTelemetry span can belong to a separate host tracer.

The server freezes the first audience snapshot. Attributes accept strings and booleans, with bounded keys/values. A lease expires at `min(config expiry, assignment expiry)`. Temporary transport/408/429/5xx refresh failure keeps a still-valid original lease, never extends it. Invalid/revoked config clears candidate use; a successful pause serves Current. Remote pause observation is bounded by the60s TTL. `unavailable` uses the explicitly supplied deployed fallback; it cannot claim that fallback is the server's latest Current.

Choosing does not record traffic. Record immediately before execution inside a Neatlogs trace; nested calls stamp its private root, and replay contexts are excluded. Complete only after the final trace of this session, preserving the original decision. Do not use an end-user ID across conversations. Current/outside-audience/outside-share/paused/fallback choices create no experiment exposure. `get_diagnostics()` reports pending/rejected receipts; `flush_async()` returns false if delivery is incomplete. Queues and caches are bounded and process-local; flush before shutdown.
