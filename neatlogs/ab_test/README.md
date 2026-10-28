# A/B assignment

Create and start a randomized test in the dashboard, then list its key in SDK initialization:

```python
import neatlogs

neatlogs.init(
    api_key="your-project-api-key",
    workflow_name="support",
    ab_test_keys=["support-model"],
)

with neatlogs.trace("answer", session_id="conversation-42"):
    version = neatlogs.ab_test(
        "support-model",
        unit="conversation-42",
        fallback={"model": "your-control-model"},
    )
    # Pass version.params to your application or model call.

neatlogs.flush()
```

For new integrations, separate choosing a version from recording the run. This prevents an abandoned choice from appearing as customer traffic. Call `record_ab_test_run` inside the trace immediately before the agent uses the selected parameters. Keep the same unit ID on every turn, export the final trace, then close the unit:

```python
version = neatlogs.choose_ab_test("support-model", unit=conversation_id, fallback={"model": "your-control-model"})
final_trace_id = None
try:
    with neatlogs.trace("answer", session_id=conversation_id) as span:
        final_trace_id = format(span.get_span_context().trace_id, "032x")
        if version.reason == "assigned" and not neatlogs.record_ab_test_run(version):
            raise RuntimeError("A/B run could not be recorded")
        answer = run_your_agent(request, version.params)
finally:
    neatlogs.flush()  # Also export an agent failure.
    if conversation_is_finished and final_trace_id and version.reason == "assigned":
        neatlogs.complete_ab_test_decision(version, final_trace_id)
        neatlogs.flush()
```

The older `ab_test` API still records immediately inside a trace for existing integrations.

Assignment uses cached settings and performs no HTTP request. Initialization fetches settings and a background thread refreshes them every 60 seconds. Missing, invalid or expired settings return control arm `a` with the supplied fallback parameters. Temporary network failures, HTTP 408/429 and server errors retain validated settings only until their original expiry; a failure never extends it. Revoked access and other invalid responses discard the cached settings immediately. Keep fallback parameters equal to your control configuration. A legacy shipped test returns its configured default version while fresh settings remain available. Staged rollouts use explicit 50% and 100% allocation epochs; paused, invalid, expired, or ended rollout settings return control.

The unit must match the trace's `session_id` for session allocation, or `end_user_id` for user allocation. Assignment inside a Neatlogs trace stamps its root with `neatlogs.ab.<key>` and queues an exposure. It does not close the unit. After every trace for a session or user has been exported, explicitly close that unit with its final trace ID:

```python
neatlogs.complete_ab_test_unit(
    "support-model",
    unit_id="conversation-42",
    arm=version.arm,
    trace_id=last_trace_id,
    allocation_version=version.allocation_version,
)
neatlogs.flush()
```

New traces after closure invalidate that unit. Do not close a user who will continue participating across sessions. Failed or pending finalization retains exposures for retry, in order; delivery stops after 24 hours. The backend also accepts eligible delayed traces until 24 hours after the experiment horizon, unless the test has already been concluded. Missing delivery or grades remain unavailable in results.

For an external feature flag, use `neatlogs.log_exposure(key, unit_id=..., arm="a" or "b")` inside the trace; an explicit `trace_id` is also supported. Close these units with `source="external"`. The backend reports that external assignment cannot be verified against its allocation hash.

`neatlogs.configure_ab_tests(keys=[...])` can fetch settings after initialization. An independent `AbTestClient(base_url=..., api_key=..., keys=[...])` supports `start()`, `assign()`, `log_exposure()`, `complete_unit()`, `flush()`, `get_diagnostics()`, and `close()`. Use one client for each project credential. A full queue or rejected exposure is reported by diagnostics and causes `flush()` to return `False`. Shutdown closes the shared cache; an independent client must be closed by its owner.

The backend copies the selected grading form and AI evaluator configuration when the experiment is created. Accepted exposure traces enter that frozen grading queue, using the normal evaluation budget and billing. Results use only answers from those grading items. Missing grades remain unavailable. AI judges need an explicit configured provider and model; a managed default cannot start a frozen comparison.

Rollout traffic is stored separately from the sealed experiment result. Each assignment returns `allocation_version`; pass it to `complete_ab_test_unit` or `log_exposure` when reporting after a configuration refresh. The client remembers the exact epoch for a previously queued trace. If that trace was assigned in multiple epochs, or its bounded history was evicted, closure without an explicit epoch returns `False`. Queued events retain their original epoch through retries. Keep the same unit identity when moving from 50% to 100%; the unchanged salt makes treatment assignment monotone.
