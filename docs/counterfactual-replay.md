# Counterfactual Trace Replay — Implementation Plan

> Naming note: "counterfactual" is the product direction (what-if analysis
> over production traces). The MVP below is honestly Phase 1 — **independent
> LLM-span probes**: each replayable LLM span is re-issued from its own
> captured messages with an explicit override. Downstream spans are never
> re-run with replayed outputs, so this is not yet causal propagation through
> a trace. The semantics below make that boundary explicit.

## 1. What exists today

**SDK shape (`neatlogs/`, v1.4.23).** Thin OpenTelemetry SDK. No custom
`Trace`/`Span` classes; spans are OTel SDK spans normalized before export.
Single source of truth for tracers is `neatlogs/_wrap_utils.py`
(`get_neatlogs_provider()`, `get_tracer()`, `get_provider_tracer()`).
The process-global OTel provider is never adopted; `neatlogs.init()` builds a
private `TracerProvider` (resource: `service.name`, `neatlogs.workflow_name`,
`user.id`, `neatlogs.tags`, `pii.*`; sampler `ParentBased(TraceIdRatioBased)`).

**Span taxonomy.** Canonical kinds in `neatlogs/core/span_kind.py`
(`WORKFLOW/AGENT/CHAIN/TOOL/RETRIEVER/EMBEDDING/GUARDRAIL/LLM/RERANKER/
VECTOR_STORE/TASK/EVALUATOR/LOG/MEMORY/MCP_TOOL`) resolved from
`neatlogs.span.kind > openinference.span.kind > traceloop.span.kind`;
`neatlogs/span_kinds/mapping.py` infers kind from span name.
`@span(kind=...)` (`neatlogs/decorators/orchestration.py` + `_base.py`) is the
custom-code entry point; `trace()` (`neatlogs/core/context.py`) is the prompt/
session context manager; `identify()` carries per-request session/end-user
(root-only `neatlogs.session.id`, `neatlogs.end_user.id|metadata`).

**LLM capture (OpenAI is best-supported).** `neatlogs/openai.py`
(`wrap_openai_client` / `wrap_async_openai_client`, plus `OpenAIInstrumentor`
for `InstrumentationManager`) emits `neatlogs.span.kind="llm"` spans named
`openai.chat.completions.create` (also `responses.*`, `parse`, embeddings,
images, audio, moderations, batches). Input convention:
`neatlogs.llm.input_messages.{i}.role|content|tool_call_id`,
`neatlogs.llm.tools.{i}.type|definition|name|description|input_schema`,
`neatlogs.llm.temperature|top_p|max_tokens|frequency_penalty|presence_penalty`,
`neatlogs.metadata`. Output via `neatlogs/core/choice_accumulator.py`
(`ChoiceAccumulator.apply()`):
`neatlogs.llm.output_messages.{i}.role|content`,
`neatlogs.llm.tool_calls.{i}.id|name|arguments|choice_index|tool_call_index`,
`neatlogs.llm.choices.{i}.finish_reason`, `neatlogs.llm.finish_reason`,
`neatlogs.llm.token_count.prompt|completion|total`,
`neatlogs.llm.metrics.duration_ms|ttft_ms`, `neatlogs.capture_fidelity`.
Sibling wrappers (`anthropic.py`, `bedrock.py`, `google_genai.py`,
`vertex_ai.py`, `openrouter.py`, `crewai.py`, `langchain.py`, …) follow the same
attribute family. `neatlogs/__init__.py:wrap()` dispatches by class/module name.

**Canonical read model.** `neatlogs/core/telemetry_v2.py:normalize_span_v2()`
converts a post-processor `ReadableSpan` into frozen `TelemetrySpanV2`
(input/output `TypedValueV2`, `semantic` per kind, usage, stream events).
`InMemoryDiagnosticSpanExporter` is the bounded in-memory capture used by
`doctor_v2.py` and available to tests. Contract frozen in
`neatlogs/contracts/v2/neatlogs-telemetry.schema.json` (+ `schema_v2.py`).

**Storage / backend.** No local trace DB. Export chain
`ObservableBatchSpanProcessor → HttpFiltering → Masking → TypedMedia →
ByteLimited → OTLPSpanExporter` POSTs gzip OTLP protobuf to
`{NEATLOGS_ENDPOINT}/v1/traces` (`https://ingest.neatlogs.com` default) with
`x-api-key`; optional `/v1/logs`. `Client` (`neatlogs/client.py`) duplicates the
pipeline per context-scoped client. Flush/shutdown semantics live in
`neatlogs/init.py`. Tests assert on `InMemorySpanExporter`
(`tests/conftest.py` sets `NEATLOGS_DISABLE_EXPORT=true`, resets OTel +
Neatlogs state per test).

**Prompts / versions.** Local `SystemPromptTemplate` / `UserPromptTemplate`
(`neatlogs/prompt/template.py`) plus backend `PromptClient`/`AsyncPromptClient`
(`neatlogs/prompt/client.py`, 60s SWR cache) with `version:int` + `labels`
(e.g. `production`); span attrs `llm.prompt_template[.version]|variables` and
`neatlogs.version` on `@span`/`trace()`. `bind_templates()` helper for
stamping compiled prompts.

**Experiments / evals.** No first-class SDK: only `kind="EVALUATOR"` spans,
`infer_span_kind_from_name` (`evaluat|score|metric`), and reserved
`datasets|evaluations|experiments` sections in the telemetry schema (future).
Prompt Playground `save-as-version` is server-side.

**MCP.** Client-side capture only: `@span(kind="MCP_TOOL")` decorator,
`openinference-instrumentation-mcp` via `InstrumentationManager.instrument_mcp`,
`openai_agents.py` maps `mcp_tools → neatlogs.span.kind=tool`. Example server
`examples/sdk_examples/shared_mcp_server.py` (FastMCP, SSE). No `MCPServer`
class in `neatlogs/`.

**AI assistant / frontend / API.** None in-repo. Dashboard assistant,
experiments UI, and trace-detail UI are server-side. No embedded HTTP server,
no `frontend/` directory. `examples/sdk_examples/` (~120 files, each with
`requirements.txt` + `.env.example`, e.g. `openai_basic.py`) and
`tests/unit/` (~48 files, `pytest testpaths=["tests/unit"]`,
`asyncio_mode=strict`) are the extension points. `benchmarks/` has one
span-processor microbench.

## 2. What is missing

1. No trace retrieval: spans are write-only (OTLP export). Nothing maps
   `trace_id → spans` locally, so `replay_trace(trace_id, …)` has no local
   source of truth.
2. No invocation reconstruction: no helper turns an LLM span's
   `neatlogs.llm.input_messages.* / tools.* / temperature…` attributes back
   into a provider call.
3. No re-execution path: no injectable LLM caller; the only call path is the
   live wrapper around the real provider SDK.
4. No replay linkage: no `replay_of / replay_id / overrides` attributes.
5. No comparison: no structured diff (only raw attribute blobs / future
   `normalize_span_v2` output).
6. No safety boundary: replaying a workflow naively would re-execute tools
   (emails, refunds, DB writes).
7. No eval handoff: `EVALUATOR` kind exists but nothing converts a
   (original, replay) pair into an eval case.

## 3. Proposed architecture

Small composable `neatlogs/replay/` package reusing existing models (no
parallel tracing, no new exporter, no server):

- `types.py` — `ReplayOverrides` (model, messages, temperature, top_p,
  max_tokens, frequency_penalty, presence_penalty + `from_dict()` with
  unknown-key rejection), `LLMRequest` / `LLMResponse`, `ReplayResult`
  (`spans: List[ReplayedSpan]`), `ReplayedSpan` snapshot, `TraceComparison`
  (+ `to_dict()`, `summary()`, and a `to_eval_case()` regression handoff),
  error types (`ReplayError`, `TraceNotFoundError`, `UnsupportedSpanError`,
  `ProviderError`), and shared `span_id()`/`trace_id()` helpers. All
  dataclasses; no new serialization framework
  (reuse `_wrap_utils.serialize` / `_base._safe_json_dumps` idioms).
- `store.py` — `TraceStore` protocol (`get_trace` / `put_trace`) +
  `InMemoryTraceStore` (dict `trace_id → list[ReadableSpan]`). Single-trace
  invariant enforced: `put_spans` rejects multi-trace span lists (exporter
  drains must be grouped by trace id first). This is an
  index over already-captured spans (e.g. drained from
  `InMemorySpanExporter.get_finished_spans()`), NOT a replacement for the
  backend. Production fetch (backend `GET /api/traces/v3/:traceId`, cf.
  `doctor_v2` probe read-back) is Phase 2 behind the same protocol.
- `extract.py` — `extract_llm_request(span) -> LLMRequest | None`:
  reads `neatlogs.llm.*` (both `neatlogs.llm.temperature` wrapper form and
  `neatlogs.llm.invocation_parameters.*` normalized form), decodes
  `input_messages` / `tools` index maps, falls back to `input.value` JSON.
  The original provider string is preserved on `LLMRequest.provider`; a
  missing provider is refused (`UnsupportedSpanError`) rather than guessed.
  Non-LLM kinds return `None`; malformed LLM spans raise `ReplayError`.
  Provider allowlist for MVP: `openai` (+ `openai`-compatible aliases:
  `azure_openai`, `openrouter`); anything else raises
  `UnsupportedSpanError`.
- `executor.py` — `replay_trace(trace_id_or_spans, overrides, *, store,
  llm_caller, tracer, allow_tool_execution=False) -> ReplayResult`.
  For each supported LLM span: build `LLMRequest`, apply overrides, call
  `llm_caller(request)`, emit a NEW span (new trace_id) via the passed or
  `get_provider_tracer()` tracer carrying original I/O + replay linkage
  attrs (`neatlogs.replay.of_trace_id/of_span_id/id/overrides`,
  `neatlogs.span.kind="llm"`, `openinference.span.kind="LLM"`).
  Unsupported kinds are skipped and recorded in `result.skipped`
  (`(span_id, kind, reason)`), never fail the whole replay.
  Default `llm_caller` is `openai_caller` (real `openai` SDK, one provider
  call per replayed LLM span billed to the caller's credentials); tests
  inject a fake. `allow_tool_execution` is accepted and forced `False` in MVP
  (accepted-but-ignored with explicit docstring + skipped-tool recording, so
  the signature is future-proof without enabling side effects).
  Resolution happens before any emission, so traces with no replayable spans
  raise without leaving stray empty traces. Replay roots carry no
  session/end-user identity (synthetic traces stay out of production session
  analytics; linkage is via `neatlogs.replay.*` attributes, which pass the
  `NeatlogsSpanProcessor` write-back and OTLP export untouched — verified
  against `attribute_processor.py`/`span_processor.py`).
- `compare.py` — `compare_traces(original, replay, overrides) ->
  TraceComparison`: match spans by `neatlogs.replay.of_span_id` else
  `(name, kind)` order; per-span field diffs (model, input, output,
  tool calls, finish reason); aggregate `latency_delta_ms`,
  `prompt/completion/total token deltas`, `cost_delta_usd` (when
  `neatlogs.llm.cost_usd` present); `changed_spans / added_spans /
  removed_spans / output_changes / tool_call_changes`. Human-readable
  `summary()` one-liner per changed span ("LLM span #…: model X→Y,
  output '…'→'…'").
- `__init__.py` — re-exports; `neatlogs/__init__.py` adds
  `replay_trace`, `compare_traces`, `ReplayOverrides`, `ReplayResult`,
  `TraceComparison`, `InMemoryTraceStore` (backwards-compatible additive
  export only).

Deviation from the prompt's sketch: `replay_trace(trace_id, overrides={...})`
  accepts either a `trace_id` (resolved via `store`) or a span list directly,
  because the SDK has no local backend read. Dict-style `overrides={...}` is
  accepted via `ReplayOverrides.from_dict()` but the canonical type is the
  dataclass, matching the repo's typed (`CachedPrompt`, `TelemetrySpanV2`)
  conventions.

## 4. Data flow

```text
capture (existing)                    replay (new)
─────────────────                     ────────────
@span / wrap(openai)                  InMemoryTraceStore.put_trace(trace_id, spans)
  → OTel spans                          ↓
  → InMemorySpanExporter (tests)      replay_trace(trace_id, overrides, llm_caller=fake)
    or OTLP backend (prod)              ↓
                                      extract_llm_request(span) per LLM span
                                        (messages/tools/params/model/provider)
                                        ↓
                                      apply ReplayOverrides (explicit only)
                                        ↓
                                      llm_caller(LLMRequest) → LLMResponse
                                        (default: openai SDK; tests: stub)
                                        ↓
                                      emit NEW spans (fresh trace_id)
                                        + neatlogs.replay.* linkage
                                        ↓
                                      ReplayResult(original_trace_id,
                                        replay_trace_id, replay_id,
                                        spans, skipped, overrides)
                                        ↓
                                      compare_traces(original, replay)
                                        → TraceComparison (structured diff)
                                        ↓
                                      (Phase 2) → EVALUATOR span / eval case
```

Original spans are never mutated (OTel `ReadableSpan` snapshots are
read-only; replay writes new spans). Replay spans flow through the normal
pipeline (masking, byte limits, delivery diagnostics) because they are real
OTel spans.

## 5. API design

```python
import neatlogs
from neatlogs import InMemoryTraceStore, ReplayOverrides

store = InMemoryTraceStore()
store.put_trace(trace_id, exporter.get_finished_spans())  # one trace per call

overrides = ReplayOverrides(model="gpt-4o-mini", temperature=0.0)
# or: ReplayOverrides.from_dict({"model": ..., "temperature": ...})

result = neatlogs.replay_trace(
    trace_id, overrides,
    store=store,
    llm_caller=my_caller,  # Callable[[LLMRequest], LLMResponse]; default = OpenAI
)
comparison = neatlogs.compare_traces(
    store.get_trace(trace_id), result.spans, overrides
)
print(comparison.summary())
result.to_dict(); comparison.to_dict()  # JSON-safe for UI / eval handoff
case = comparison.to_eval_case()  # rejected vs preferred output + provenance
```

`LLMRequest{provider, model, messages[{role,content}], tools[{name,
definition}], temperature, top_p, max_tokens, frequency_penalty,
presence_penalty}` →
`LLMResponse{content, tool_calls[{name, arguments, id}], model,
prompt_tokens, completion_tokens, total_tokens, latency_ms, finish_reason}`.
Tool calls in the response are recorded as span data only — never executed
(`allow_tool_execution=False` enforced).

## 6. Security concerns

- **Tool side effects (primary risk).** MVP replays LLM spans only and never
  executes tools/external APIs. `allow_tool_execution` defaults to `False`
  and is refused in MVP; skipped tools are recorded explicitly. Documented in
  module docstring + example.
- **Secrets / API keys.** Span attributes never carry provider keys; replay
  uses the caller's own credentials at call time. `neatlogs.replay.overrides`
  stores only model/prompt/config deltas, never secrets.
- **PII.** Replay re-sends original inputs (user messages, retrieval context)
  to the provider. Callers must apply the existing `mask=` boundary or
  server-side PII redaction before replaying production traces; the replay
  path emits through the same `MaskingSpanExporter` pipeline. Documented as a
  caller obligation.
- **Authorization.** No new network surface in MVP (no backend fetch, no MCP
  tools, no UI endpoint). Backend trace fetch (Phase 2) must reuse the
  existing `x-api-key` + project scoping, never a separate auth scheme.
- **Provenance.** Every replay span carries `of_trace_id/of_span_id/replay_id`
  so replays are distinguishable from production in the backend.

## 7. Determinism / replay limitations

- LLM output is non-deterministic (temperature 0 narrows, does not eliminate).
  Same input ≠ same output; comparison must be read as one sample, not proof.
  Replaying with empty overrides is a supported determinism baseline.
- **Fan-out, not chained.** MVP replays every LLM span from its *original*
  captured messages. In an `LLM → tool → LLM → tool → LLM` trace, all three
  LLM spans are re-issued as independent probes: LLM2 still sees the original
  tool result, not the replayed one. The "different tool call → correct
  answer" story holds end-to-end only for single-LLM traces; for multi-step
  agents each probe answers "what would *this step* have done with the new
  prompt?". Chained re-execution (feeding replayed outputs + stubbed tool
  results forward) is Phase 2.
- MVP replays **single LLM spans**, not whole workflows: no agent loop,
  no retriever re-query (recorded retrieval context is carried as span data,
  not re-fetched), no DB/clock freezing, no streaming-event fidelity.
  Sibling-span context (retriever docs, tool results outside message history)
  is not merged into replayed messages.
- Timestamps, `span_id`/`trace_id`, token counts, latency are inherently new.
- Retrieval context, tool schemas, and prior messages are preserved as
  captured data, which may be stale relative to current production state —
  reported explicitly rather than hidden.
- Cost: one provider call (and charge) per replayed LLM span.
- Full workflow replay (re-running `@span`-decorated functions with stubbed
  tools) is Phase 2 and requires a tool-stubbing contract that does not exist
  yet.

## 8. MVP scope

1. `neatlogs/replay/` package (types, store, extract, executor, compare,
   OpenAI default caller) + top-level re-exports.
2. LLM-span replay for OpenAI (`openai` provider family); other providers →
   `UnsupportedSpanError` recorded in `skipped`, not fatal.
3. New-trace emission with `neatlogs.replay.*` linkage via existing span
   metadata mechanisms.
4. Structured `TraceComparison` (changed/added/removed spans, output + tool
   changes, latency/token/cost deltas, `summary()`).
5. `tests/unit/test_counterfactual_replay.py`: successful replay, override
   applied, original unchanged, replay references original, comparison detects
   changed output + changed tool call, malformed/missing trace,
   unsupported span type/provider, provider failure, mixed-trace rejection,
   empty-provider refusal, penalty round-trip, `to_eval_case` (+ its
   no-output-change refusal).
6. `examples/sdk_examples/counterfactual_replay_basic.py`: offline runnable
   (fake caller, no API keys) — build trace → store → replay with changed
   prompt → compare → print summary. Spans are nested explicitly on one
   tracer: after `init()` the `@span` decorator threads parents privately,
   so a hand-rolled LLM span would fragment into its own trace; in
   production the LLM span comes from `wrap()` and nests automatically.
   Follows `openai_basic.py` structure.
7. Docs: this file. No UI, no MCP tools, no eval runner. The eval handoff is
   `TraceComparison.to_eval_case()` (plain JSON: input text, rejected vs
   preferred output, overrides, provenance) — no second evaluation system.

## 9. Future extensions

- **Phase 2a — retrieval:** backend fetch `GET /api/traces/v3/:traceId`
  (same pattern as `doctor_probe_v2` read-back) behind `TraceStore`;
  server-side ingestion of `to_eval_case()` payloads into the existing
  Playground dataset / experiments flow (the payload shape is deliberately
  plain JSON so the server defines the final schema).
- **Phase 2b — UI:** minimal "Replay & Compare" entry on the existing trace
  detail view reusing trace components; override pickers for prompt/model;
  side-by-side original vs replay (consumes `TraceComparison.to_dict()`).
- **Phase 2c — MCP:** `replay_trace` / `compare_traces` tools only if they
  fit the existing `instrument_mcp` client-side pattern without a new server.
- **Phase 2d — broader replay:** Anthropic/Gemini callers, multi-span traces,
  workflow re-execution with explicit tool stubs, retrieval-context freezing,
  deterministic sampling controls.
