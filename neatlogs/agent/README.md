# Agent connection (v2)

Import `AgentClient` from `neatlogs.agent`. This adapter uses the server's persistent enrollment for Customer ID (`end_user`) or Conversation ID (`session`) units. The server freezes the assignment unit and completion policy; this adapter does not make rollout decisions. Existing `neatlogs.ab_test` remains protocol v1.

```python
from neatlogs.agent import AgentClient
import neatlogs
import asyncio

agents = AgentClient(base_url=base_url, api_key=api_key, keys=["support"])
# Inside your async handler, before opening the root trace:
version = await agents.choose_agent("support", unit=customer_id,
    attributes={"environment": "production"}, fallback={"model": deployed_model})
trace_id = None
try:
    with neatlogs.trace("reply", session_id=conversation_id, end_user_id=customer_id) as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        recorded = agents.record_agent_run(version)  # Record every actual execution.
        if version.reason == "assigned" and not recorded:
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

Reuse one client. `choose_agent` performs bounded HTTP when config or the customer's or conversation's lease needs renewal; it runs blocking transport in a worker thread. `refresh_async()` explicitly refreshes config. Optional synchronous `start()` uses the existing daemon refresh/retry loop; do not call a blocking startup method on an event loop. Use the yielded Neatlogs span for trace identity, since the global OpenTelemetry span can belong to a separate host tracer.

The server freezes the first audience snapshot. Attributes accept strings and booleans, with bounded keys/values. A lease expires at `min(config expiry, assignment expiry)`. Temporary transport/408/429/5xx refresh failure keeps a still-valid original lease, never extends it. Invalid/revoked config clears candidate use; a successful pause serves Current. Remote pause observation is bounded by the60s TTL. `unavailable` uses the explicitly supplied deployed fallback; it cannot claim that fallback is the server's latest Current.

Choosing does not record traffic. Record immediately before execution inside a Neatlogs trace; nested calls stamp its private root, and replay contexts are excluded. Pass the customer ID as `unit` and `end_user_id` for customer assignment, while `session_id` remains the actual conversation ID. For conversation assignment, use the conversation ID as `unit` and `session_id`. After exporting the final trace of each conversation, call `complete_agent_decision` with its original decision. It sends `conversationComplete: true`. Session/`explicit` assignment also sends `unitComplete: true`; customer/`fixed_horizon` assignment leaves the customer open until the server cutoff and delivery grace, so later conversations can use the same active assignment. Current/outside-audience/outside-share/paused/fallback choices create no experiment exposure. `get_diagnostics()` reports pending/rejected receipts; `flush_async()` returns false if delivery is incomplete. Queues and caches are bounded and process-local; flush before shutdown.

The client requests and validates `serving-trace-v1` acknowledgement on the existing endpoints. Legacy servers/policies keep their existing receipt behavior without claiming enhanced historical coverage. A compatible deployed SDK/source build is required for publications; execute the returned params. Published decisions validate the advertised publication/fingerprint pair, serving revision and relevant raw audience attributes, and never create an AB receipt.

Always call `record_agent_run` inside the root for every actual execution, including fallback, Current and published choices. It writes ordinary agent/unit/reason/actual-fingerprint metadata; the boolean still reports whether a signed AB receipt was queued, so nonassigned decisions normally return false. Optional `decision.cause` and root `decision_cause` contain only `transport_unavailable`, `invalid_identity` or `unsupported_canonical_params`; unknown causes are omitted. Mutating params before recording rejects new-profile signed/publication identity. No raw exception or credential data is captured.

Canonical identity reuses the installed RFC8785 helper and NFC-normalizes parameter string values only. Keys and audience strings are unchanged. Python runtime Unicode data can lag the server; unknown normalization characters, lone surrogates and native integers that cannot be represented losslessly fail closed for new-profile evidence. For example, a default wire parse of `1000000000000000100` cannot safely stand in for the server's binary64 value. Do not coerce that integer or copy the advertised hash. Legacy policy execution remains available; ordinary unsupported identity remains explicit and cannot support enhanced experimental coverage. Entirely unexported executions remain an instrumentation assumption.

Update every instance that serves this agent before starting a new test.

A true recorder result means a signed receipt was queued; it does not mean final telemetry was accepted or primary cohort coverage is complete. Signed identity keeps priority over ordinary tags near attribute limits; known missing signed tags are not queued.
