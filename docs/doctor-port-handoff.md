# Trace Doctor — TS & Go Port Handoff

> **Audience:** Engineers porting the `neatlogs-doctor` CLI to TypeScript and Go.
> **Goal:** Reimplement `neatlogs/doctor.py` (~2,000 lines, 38 top-level functions across PR #20 + PR #21) and its full test suite (142 unit tests + 17 integration tests + 4 E2E demo scripts) in your target language, **with no behavior change vs. the Python reference**.
> **Reference implementation:** `neatlogs/doctor.py` in this repo, branch `feat/trace-doctor @ b5f8373` (PR #20 + PR #21 + E2E). All `path:line` references below are into that file.
> **Test reference:** `tests/unit/test_doctor.py` (2,577+ lines), `tests/integration/test_doctor_e2e.py` (617 lines), `docs/pr20-evidence/doctor_e2e_{demo,dimensions,realsdk,pr21}.py`.

## What's in this handoff (PR #20 + PR #21)

**PR #20 (Trace Doctor v1):** 14 finding codes across 4 dimensions — hierarchy pathologies (orphan-parent, self-parent, duplicate-span-id, multiple-roots, cycle), missing I/O on LLM/tool/retriever spans, foreign-instrumentation detection, agent-without-llm, and the 4 new dimensions (init-after-client, missing-span-kind, data-integrity trio, pipeline-stage-summary). LLM actionability fields (`fix_class`, `automated_fix_available`, `doc_url`, `related_codes`).

**PR #21 (Trace Doctor v2):** 6 more finding codes — OTel GenAI semconv compliance (`otel-genai-missing`, `otel-genai-inconsistent`) for interoperability with Langfuse/Phoenix/Arize, token-waste patterns (`oversized-prompt`, `repeated-system-prompt`, `unused-tool-definition`) for cost/latency debugging, and the `--emit-fix <code>` CLI flag for manual-fix snippets. PII-gated `repeated-system-prompt` behind `--read-prompt-content` (default off).

**Total: 20 finding codes, 9 fix_class values, 4 pipeline stages, 7 CLI flags, 3 exit codes (0/1/2).**

---

## 1. Scope: what you're reimplementing

The Trace Doctor is a **local, read-only linter** for span log files written by the Neatlogs log exporter. It runs as a CLI (`neatlogs-doctor path/to/file.log [--json] [--run-id ID] [--foreign-only]`) and outputs a list of findings — never reaches the network, never mutates input.

**What it does:**
1. Reads a JSONL span log file (one span per line).
2. Groups spans by run (`session.id` attr → `trace_id` fallback → sentinel).
3. For each trace, runs 12 independent checks; aggregates per-run; emits a `pipeline-stage-summary` when one stage dominates.
4. Returns a `DoctorReport` with: `path`, `spans_read`, `trace_count`, `run_count`, `invalid_lines`, `findings: tuple[DoctorFinding, ...]`.
5. CLI serializes via `format_report()` (human) or `--json` (machine).

**What it does NOT do:** no network calls, no SDK initialization, no span emission. The doctor is offline-only.

---

## 2. Span log format (the input)

Every line of the input JSONL is a span dict. The doctor reads these keys; missing keys are tolerated (None / empty defaults). All values are JSON-serializable.

### 2.1 Required keys
| Key | Type | Notes |
|---|---|---|
| `trace_id` | string | Empty string allowed (treated as a real trace). |
| `span_id` | string | Used as a node identifier in hierarchy checks. |
| `parent_span_id` | string \| null | null for root spans. |
| `name` | string | Span display name. |
| `kind` | string | One of `workflow`, `chain`, `agent`, `tool`, `llm`, `embedding`, `reranker`, `retriever`, `http`, `mcp_tool`, or anything else (mapped to `UNKNOWN`). |

### 2.2 Optional keys
| Key | Type | Default if missing | Notes |
|---|---|---|---|
| `start_time` | int (ns epoch) | none | Used for latency-mismatch check. |
| `end_time` | int (ns epoch) | none | Used for zero-duration + latency-mismatch checks. |
| `duration_ns` | int | none | If present, preferred over `end_time - start_time` for the zero-duration check. |
| `status` | dict | `{"code": "OK"}` | See §2.3. |
| `events` | list of dicts | `[]` | Used for exception-event detection. |
| `attributes` | dict | `{}` | Most checks look here. |
| `instrumentation_scope` | dict | none | Used for foreign-instrumentation detection. See §2.4. |
| `session.id` | string (in `attributes`) | `trace_id` fallback | Used for run grouping. |

### 2.3 Status format (tolerant — accept BOTH)
The Neatlogs log exporter normalizes to `{"code": "ERROR", "description": "..."}`. Some other exporters (e.g. raw OTel SDK, foreign SDKs) emit `{"status_code": {"name": "ERROR", "value": 2}}`. **Accept both.** The Python reference has `_span_status_is_error()` for this; replicate that logic.

Helper logic:
```python
def span_status_is_error(status: dict) -> bool:
    if not isinstance(status, dict): return False
    code = status.get("code")
    if isinstance(code, str) and code.upper() in ("ERROR", "ERROR_STATUS"):
        return True
    sc = status.get("status_code")
    if isinstance(sc, dict):
        name = sc.get("name")
        if isinstance(name, str) and name.upper() in ("ERROR", "ERROR_STATUS"):
            return True
    if isinstance(sc, str) and sc.upper() in ("ERROR", "ERROR_STATUS"):
        return True
    return False
```

### 2.4 Instrumentation scope format
Two acceptable forms:
- `{"name": "neatlogs.core.context", "version": "1.4.20"}` (modern)
- `"neatlogs.core.context"` (string, legacy)

The doctor recognizes names starting with `neatlogs.` as own-scope; everything else is foreign.

### 2.5 Event format
Events are dicts. The doctor checks `event["name"] == "exception"`. Both `"name"` and canonical `"exception"` event names from OTel SDK are supported. A more robust check is to also look for `event["attributes"]["exception.type"]` — but the reference implementation only checks `name == "exception"`. **For the TS/Go port, use the reference behavior; you can add the attribute-based fallback as an enhancement.**

---

## 3. Finding codes (the 20 output types)

Every output is a `DoctorFinding` with these required fields:
- `severity: "error" | "warning" | "info"`
- `code: string` (one of the 20 below)
- `title: string` (one-line, human-readable)
- `evidence: string` (what was found; max 200 chars, longer evidence gets `...` suffix — see `_truncate()`)
- `suggestion: string` (how to fix; can be long)
- `trace_id: string | null` (the trace the issue is in, or null for run-level findings)
- `run_id: string | null` (the run, or null for trace/file-level)
- `fix_class: string | null` (one of the 8 below; for LLM/coding-agent consumption)
- `automated_fix_available: bool` (default false; True if a coding agent could fix without a human)
- `doc_url: string | null` (a path the consumer can read for more context — see §11)
- `related_codes: tuple[string, ...]` (codes commonly seen together; the consumer can cross-reference)

### 3.1 The 20 finding codes (PR #20 + PR #21)

| Code | Severity | fix_class | auto-fix | Description | Source |
|---|---|---|---|---|---|
| `rootless-http-only` | warning | (none) | False | All visible spans are rootless HTTP — auto-instrumented requests without a parent. | `_is_rootless_http_only()` |
| `missing-root-kind` | warning | (none) | False | No root span of kind workflow/chain/agent/mcp_tool. | `_diagnose_trace()` |
| `orphan-parent` | warning | (none) | False | `parent_span_id` references a span_id that doesn't exist. | `_orphan_parent_findings()` |
| `self-parent` | error | (none) | False | `span_id == parent_span_id`. Reported separately from `cycle` (which is filtered out for self-cycles). | `_self_parent_findings()` |
| `duplicate-span-id` | error | (none) | False | Same `span_id` appears twice in the same trace. | `_duplicate_span_id_findings()` |
| `multiple-roots` | warning | (none) | False | More than one root span (no `parent_span_id`) in a single trace. | `_multiple_roots_findings()` |
| `cycle` | error | (none) | False | Back-edge in the span parent/child tree. | `_cycle_findings()` |
| `agent-without-llm` | warning | (none) | False | An `agent` span has no `llm` descendant in its subtree. | `_agent_without_llm_findings()` |
| `llm-missing-io` | warning | `capture` | False | An `llm` span is missing input or output attributes. | `_missing_io_findings()` |
| `otel-genai-missing` | warning | `config` | False | An LLM-kind span (neatlogs `kind=llm` OR `gen_ai.operation.name` ∈ {chat, text_completion, generate_content}) lacks `gen_ai.operation.name`. Trace won't be interoperable with OTel GenAI tools (Langfuse, Phoenix, Arize). | `_otel_genai_findings()` |
| `otel-genai-inconsistent` | info | `config` | False | A span has BOTH `neatlogs.span.kind=llm` AND `gen_ai.operation.name` but they disagree (e.g. neatlogs says "llm" but OTel says "embeddings"). Wrapper bug or migration in progress. | `_otel_genai_findings()` |
| `oversized-prompt` | warning | `config` | False | A single LLM span's prompt content exceeds 50,000 characters. Almost certainly a bug (leaked retrieved document, CSV, or log dump). | `_token_waste_findings()` |
| `repeated-system-prompt` | info | `config` | False | The same system prompt content appears 10+ times across LLM spans in the same trace. PII concern — only fires when `--read-prompt-content` is on (default off). | `_token_waste_findings()` |
| `unused-tool-definition` | info | `config` | False | A tool defined on an LLM span is never called in any subsequent span. No PII concern (tool names only). | `_token_waste_findings()` |
| `tool-missing-io` | warning | `capture` | False | A `tool` span is missing input or output attributes. | `_missing_io_findings()` |
| `retriever-missing-io` | warning | `capture` | False | A `retriever` span is missing input or output attributes. | `_missing_io_findings()` |
| `foreign-instrumentation-detected` | warning | (none) | False | Spans from a non-`neatlogs` scope are present (e.g. openlit, langfuse co-tenant). | `_foreign_instrumentation_findings()` |
| `multi-run-log` | warning | (none) | False | The log file contains spans from more than one `session.id`. Suggests using `--run-id` to scope. | `_diagnose_trace()` |
| `run-id-not-found` | error | (none) | False | `--run-id X` was passed but no spans have that `session.id`. | `_diagnose_trace()` |
| `file-not-found` | error | (none) | False | The input path doesn't exist. | `_diagnose_trace()` |
| `no-spans` | error | (none) | False | The file has 0 valid spans (might be empty or all-invalid). | `_diagnose_trace()` |
| `invalid-jsonl` | error\|warning | (none) | False | The file has unparseable JSON lines. | `_diagnose_trace()` |
| `scope-not-preserved` | info | (none) | False | Spans lack `instrumentation_scope` (backward compat — older SDK versions didn't emit it). | `_diagnose_trace()` |
| **`init-after-client`** | **error** | `init_order` | **True** | **A span has no Neatlogs init markers (e.g. `neatlogs.instrumentation.name`) — the wrapper was created before `neatlogs.init()`. Auto-fixable: move init to the top of the entry point.** | `_init_order_findings()` |
| **`missing-span-kind`** | **warning** | `attribute` | **False** | **Some spans lack `neatlogs.span.kind`. Dashboard will mis-categorize them. Suppressed when ALL spans miss kind (that's the init-order symptom).** | `_attribute_completeness_findings()` |
| **`zero-duration-span`** | **warning** | `data_integrity` | **False** | **Some spans have `duration_ns == 0` (or `start_time == end_time`). Wrapper likely crashed before `span.end()`.** | `_data_integrity_findings()` |
| **`error-status-no-event`** | **warning** | `data_integrity` | **False** | **Some spans are marked ERROR but lack an `exception` event. Dashboard's error view will be empty.** | `_data_integrity_findings()` |
| **`latency-mismatch`** | **error** | `data_integrity` | **False** | **Some spans have `end_time < start_time` (clock issue — different sources for start and end).** | `_data_integrity_findings()` |
| **`pipeline-stage-summary`** | **info** | `pipeline` | **False** | **A run-level finding that fires when one stage (init/instrument/span/hierarchy) has >50% of findings. Title and suggestion are stage-specific (not hardcoded to "init").** | `_pipeline_stage_run_finding()` |

**Bold rows = the 4 new diagnostic dimensions added in PR #20.** All other codes were pre-existing.

### 3.2 Fix-class taxonomy
The 9 fix_class values map to 4 pipeline stages via `_FIX_CLASS_TO_STAGE`:

| Stage | fix_class |
|---|---|
| `init` | `init_order`, `config`, `pipeline` |
| `instrument` | `instrumentation`, `capture` |
| `span` | `data_integrity`, `attribute` |
| `hierarchy` | `hierarchy` |

Note: `config` is shared between `init` (env-var / init-order issues) and the new token-waste / OTel GenAI cluster. The 5 PR #21 codes (`otel-genai-missing`, `otel-genai-inconsistent`, `oversized-prompt`, `repeated-system-prompt`, `unused-tool-definition`) all use `config` and therefore cluster at the `init` stage for the pipeline-stage summary. This is intentional — they describe "what the user must configure or how the wrapper is registered" rather than "what's wrong with the captured span data".

The `pipeline-stage-summary` finding's `related_codes` field filters by the dominant stage's fix_class values — see §6 for the exact logic and the bug fix history.

---

## 4. Diagnostic logic — exact rules

This is the testable behavior. Each subsection is one finding's predicate.

### 4.1 `rootless-http-only` (warning, no fix_class)
**Predicate:** All visible spans in the trace have `kind == "http"` AND no `parent_span_id`. Returns ONE finding per affected trace.
**Source:** `_is_rootless_http_only()` (lines 1115–1120). Triggers an early-return: when this fires, only `init-after-client`, `missing-span-kind`, and data-integrity findings still run (see §6.1 ordering).

### 4.2 `missing-root-kind` (warning, no fix_class)
**Predicate:** The set of root span kinds has no intersection with `ROOT_KINDS = {"workflow", "chain", "agent", "mcp_tool"}`. One finding per trace.
**Source:** `_diagnose_trace()` lines 571–586. **Skipped if `rootless-http-only` already fired (early-return).**

### 4.3 `orphan-parent` (warning, no fix_class)
**Predicate:** A visible span has `parent_span_id` set but no visible span has that `span_id`. Lists up to 3 examples in evidence.
**Source:** `_orphan_parent_findings()` lines 652–681.

### 4.4 `self-parent` (error, no fix_class)
**Predicate:** A visible span has `span_id == parent_span_id`. Only reports the **first** such span per trace (one finding per trace, not per span). Self-cycles are filtered out of the regular `cycle` walk separately.
**Source:** `_self_parent_findings()` lines 683–708. **Important interaction with `cycle`:** if a span self-cycles, it's reported as `self-parent` and the `cycle` walker skips it (filtered at top of `_cycle_findings()`).

### 4.5 `duplicate-span-id` (error, no fix_class)
**Predicate:** A `span_id` appears more than once in the visible spans of a single trace. Lists up to 3 examples.
**Source:** `_duplicate_span_id_findings()` lines 710–736.

### 4.6 `multiple-roots` (warning, no fix_class)
**Predicate:** More than 1 root span (`parent_span_id` is null/missing) in a trace. Lists up to 3 root span IDs in evidence.
**Source:** `_multiple_roots_findings()` lines 738–762.

### 4.7 `cycle` (error, no fix_class)
**Predicate:** A back-edge in the parent/child tree, where a node reaches a node in its own ancestor path.
**Algorithm:** Iterative DFS from each unvisited node. Three sets: `done` (fully explored), `in_path` (currently on the DFS stack). Walk DOWN via `child_map`. Back-edge detected when child id is in `in_path` (NOT `done`). O(V+E).
**Self-parent handled separately** (filtered out before the DFS walk).
**Performance:** 5K spans: 18ms; 10K: 37ms; 50K: 270ms; 100K: 575ms. Linear scaling.
**Source:** `_cycle_findings()` lines 764–885.

### 4.8 `agent-without-llm` (warning, no fix_class)
**Predicate:** An `agent` span has no `llm` descendant in its subtree. Subtree-walked via `child_map` (per-agent, not per-trace). Reports up to 3 agent names.
**Source:** `_agent_without_llm_findings()` lines 887–926. **Skipped if `rootless-http-only` already fired.**

### 4.9 `llm-missing-io` / `tool-missing-io` / `retriever-missing-io` (warning, fix_class=`capture`)
**Predicate:** An `llm`/`tool`/`retriever` span lacks input or output attributes.
- **llm:** Input = `neatlogs.llm.input_messages.*` OR `neatlogs.llm.prompts.*` OR `neatlogs.llm.system` (system-only counts as input). Output = `neatlogs.llm.output_messages.*`. Content must be **non-empty** for the test to pass (this is the "Bug #1" from PR #20).
- **tool:** Input = `neatlogs.tool.parameters` (must JSON-decode and be non-empty `{}`). Output = `neatlogs.tool.output` (must JSON-decode and be non-empty `{}`).
- **retriever:** Input = `neatlogs.retriever.query` (non-empty string). Output = `neatlogs.retriever.documents` (non-empty list/dict with content).
**Suppressed** when the trace has no `llm`/`tool`/`retriever` spans (to avoid false positives on traces that never did those operations). See `_missing_io_suppression_when_no_io_kinds_in_trace`.
**Source:** `_missing_io_findings()` lines 615–650. **Skipped if `rootless-http-only` already fired.**

### 4.10 `foreign-instrumentation-detected` (warning, no fix_class)
**Predicate:** Any visible span has `instrumentation_scope.name` not starting with `neatlogs.`. Lists up to 3 scope names in evidence.
**Dedup:** If the same foreign scope appears in multiple runs, the dedup is by scope name (not per-trace).
**Source:** `_foreign_instrumentation_findings()` lines 928–985. Always runs (no early-return skips this).

### 4.11 `init-after-client` (error, fix_class=`init_order`, auto-fixable=True) — NEW
**Predicate:** A span exists BUT none of the init-marker keys are in its attributes.
**Init markers:** `neatlogs.instrumentation.name`, `neatlogs.span.kind`, `neatlogs.workflow_name`.
**Behavior:** Emits ONE finding per trace (the first span without markers — the rest are downstream of the same root cause).
**Auto-fixable because:** moving `neatlogs.init()` to the top of the entry point is mechanical.
**Suggestion text:** "Move neatlogs.init() to the very top of your entry point, BEFORE constructing any LLM client. If you cannot reorder, call neatlogs.shutdown() then neatlogs.init() again."
**Source:** `_init_order_findings()` lines 1157–1202.
**Related codes:** `("no-spans", "missing-root-kind")` (the symptoms that the symptom causes).
**Doc URL:** `skills/neatlogs/references/troubleshooting.md#1-import-order-issues-most-common-mistake`.

### 4.12 `missing-span-kind` (warning, fix_class=`attribute`, auto-fixable=False) — NEW
**Predicate:** A span lacks `neatlogs.span.kind` in its attributes.
**Suppressed when:** ALL spans miss the kind (that's the init-order symptom; init-after-client fires instead). Suppressed also when zero spans miss the kind.
**Behavior:** ONE finding per trace. Lists up to 3 example span names.
**Source:** `_attribute_completeness_findings()` lines 1205–1251.
**Related codes:** none.
**Doc URL:** `skills/neatlogs/references/troubleshooting.md#6-common-anti-patterns-table`.

### 4.13 Data-integrity findings (warning/error, fix_class=`data_integrity`, auto-fixable=False) — NEW
**Three sub-checks per span, run in one pass:**

| Sub-code | Severity | Predicate |
|---|---|---|
| `zero-duration-span` | warning | `duration_ns == 0` OR (`start_time == end_time` when both are numbers) |
| `error-status-no-event` | warning | `_span_status_is_error(status)` AND no event with `name == "exception"` |
| `latency-mismatch` | error | `start_time > end_time` (both numbers) |

**Internal spans** (with `neatlogs.internal=True` attribute, or named `neatlogs.trace.complete`) are EXCLUDED from these checks.
**Behavior:** ONE finding per sub-check per trace. Lists up to 3 example span names in evidence.
**Related codes:** `zero-duration-span` ↔ `error-status-no-event` (commonly co-occur).
**Source:** `_data_integrity_findings()` lines 1254–1357. Always runs (no early-return skips this; that's the bug we fixed in `9669556`).

### 4.14 `pipeline-stage-summary` (info, fix_class=`pipeline`, auto-fixable=False) — NEW
**Build rule:** Count findings per stage via the `_FIX_CLASS_TO_STAGE` map. If a single stage has more findings than all others combined (`stage_count * 2 > total`), the dominant stage triggers. Title and suggestion are stage-specific.
**Suppressed** when: no findings cluster at a single stage.
**Skipped** under `--foreign-only` (foreign findings don't represent a pipeline failure on the user's side).
**Stage-specific suggestions (the `_STAGE_SUGGESTIONS` dict):**
- `init`: "Move neatlogs.init() to the top of the entry point (before any client is constructed), then re-run the doctor — the rest usually resolves once init is right."
- `instrument`: "Most findings are about wrappers not capturing or being reached. Verify the LLM client is constructed after neatlogs.init() and that the wrapper registered for the framework is actually installed."
- `span`: "Most findings are about the captured span data itself. Check the wrapper's end() and exception-recording paths; a crashed wrapper leaves spans with zero duration and no events."
- `hierarchy`: "Most findings are about parent/child relationships. Verify that each wrapper sets parent_span_id correctly; duplicates and orphan parents usually mean a wrapper is creating spans outside the active context."
**Source:** `_pipeline_stage_run_finding()` lines 1412–1448.

### 4.15 Run-level and file-level findings
- `multi-run-log` (warning): fires when `len(runs) > 1` and `run_id` is None. Evidence: "N runs detected, M spans total. Pass --run-id <id> to scope the report to one run."
- `run-id-not-found` (error): fires when `run_id` is set but not in `runs`. Evidence: "run_id='X' but log has runs: [list]".
- `file-not-found` (error): fires when the path doesn't exist.
- `no-spans` (error): fires when the file has 0 valid spans.
- `invalid-jsonl` (error/warning): fires when the file has unparseable JSON lines. Error if no valid spans, warning otherwise. Evidence lists up to 5 line numbers.
- `scope-not-preserved` (info): fires when ALL spans lack `instrumentation_scope`. Backward compat — older SDK versions didn't emit it. Evidence: "All N span(s) lack instrumentation_scope. Foreign-instrumentation detection can't run on this file."

### 4.16 PR #21 additions — OTel GenAI semconv + token-waste + manual-fix snippets

#### A. OTel GenAI semantic-convention findings

These fire when an LLM-kind span doesn't carry the OTel GenAI semconv attributes. The point is interoperability: Langfuse, Phoenix, Arize, and other OTel GenAI consumers filter on `gen_ai.*` and will skip spans without them.

**The shared LLM-kind predicate** (`_is_llm_kind(span)`):
- `attrs["neatlogs.span.kind"] == "llm"`, OR
- `attrs["gen_ai.operation.name"]` is a string in `OTEL_GENAI_LLM_OPERATIONS = {"chat", "text_completion", "generate_content"}`

**`otel-genai-missing` (warning, fix_class=`config`, auto-fixable=False)**
**Predicate:** A span passes `_is_llm_kind` AND `attrs["gen_ai.operation.name"]` is None. ONE finding per trace; evidence lists `N LLM span(s) missing gen_ai.operation.name`.
**Counts span-kinds internally** so the finding fires once even if multiple LLM spans miss the op-name (the evidence text is just the count, not per-span names).
**Internal spans are excluded.**
**Suppressed** when no spans are LLM-kind.
**Source:** `_otel_genai_findings()` lines 1273–1347.

**`otel-genai-inconsistent` (info, fix_class=`config`, auto-fixable=False)**
**Predicate:** A span has `neatlogs.span.kind == "llm"` AND `gen_ai.operation.name` is a string AND that op is NOT in `OTEL_GENAI_LLM_OPERATIONS`. ONE finding per offending span (not deduplicated) — list up to 3 example span names in evidence.
**Why per-span:** the inconsistency is a per-span wrapper bug; the user needs the span name to find it.
**Related codes:** `("missing-span-kind",)` — cross-reference.
**Source:** `_otel_genai_findings()` lines 1301–1323.

**The 6 OTel GenAI attribute keys** (read in this order; OTel takes precedence, neatlogs is the fallback):
- `gen_ai.operation.name` (string) — required for `otel-genai-missing` to NOT fire.
- `gen_ai.provider.name` (string) — optional, used in healthy-trace fixtures.
- `gen_ai.request.model` (string) — optional, used in healthy-trace fixtures.
- `gen_ai.usage.input_tokens` (int) — optional, used in healthy-trace fixtures.
- `gen_ai.usage.output_tokens` (int) — optional, used in healthy-trace fixtures.
- `gen_ai.response.finish_reasons` (list of string) — optional, used in healthy-trace fixtures.
- `gen_ai.input.messages` (list of `{role, content}` dicts) — used for prompt size + system prompt.
- `gen_ai.system_instructions` (list of `{type, content}` dicts) — used for system prompt.
- `gen_ai.tool.definitions` (list of `{name, ...}` dicts) — used for tool definitions.
- `gen_ai.output.messages` (list of `{role, content, finish_reason, tool_calls}` dicts) — used for tool calls.

Reference: <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md>

#### B. Token-waste findings

**`oversized-prompt` (warning, fix_class=`config`, auto-fixable=False)**
**Predicate:** A single LLM span's prompt content exceeds 50,000 characters (constant `OVERSIZED_PROMPT_CHAR_THRESHOLD`).
**Always runs** — no PII concern; only counts characters, doesn't read content.
**Counts chars via `_llm_prompt_size(span)`** which walks FOUR locations:
- `gen_ai.input.messages[*].content` — string OR list of `{type, text}` parts.
- `neatlogs.llm.input_messages.*` — each numbered attribute is a serialized message string.
- `neatlogs.llm.prompts.*` — same pattern (older neatlogs layout).
- `neatlogs.llm.system` — single string.

The OTel locations are walked first (because they include the system message as part of `input.messages`). The neatlogs `system` attr is added to the total but is NOT double-counted if it's already in the OTel list (the OTel walk handles that; neatlogs `system` is the standalone fallback for traces that lack OTel attrs).
**ONE finding per trace**; evidence lists up to 3 example span names with their prompt sizes.
**Internal spans are excluded.**
**Source:** `_token_waste_findings()` lines 1738–1801, helper `_llm_prompt_size()` lines 1600–1635.

**`repeated-system-prompt` (info, fix_class=`config`, auto-fixable=False)**
**Predicate:** The same system prompt content appears 10+ times across LLM spans in the trace (constant `REPEATED_SYSTEM_PROMPT_THRESHOLD`).
**PII-gated** — only fires when `read_prompt_content=True` (default off, opt-in via `--read-prompt-content`). When off, the check is silently skipped. This is the only PII-sensitive check in the doctor.
**System-prompt source** (`_llm_system_prompt(span)`): prefers `neatlogs.llm.system` (string), falls back to the first `gen_ai.system_instructions[*].content` (string OR list of `{type, text}` parts, joined with `\n`). Returns None if neither exists.
**ONE finding per trace**; evidence lists the count of distinct system prompts that crossed the threshold and the top repeat count + char length.
**Source:** `_token_waste_findings()` lines 1803–1832, helper `_llm_system_prompt()` lines 1638–1664.

**`unused-tool-definition` (info, fix_class=`config`, auto-fixable=False)**
**Predicate:** A tool defined on an LLM span's `gen_ai.tool.definitions` or `neatlogs.llm.tools` is never called in any subsequent span.
**Always runs** — no PII concern (tool names only).
**Tool-definition source** (`_llm_tool_definitions(span)`):
- OTel: `gen_ai.tool.definitions[*].name`.
- neatlogs: `neatlogs.llm.tools` is a JSON string of `[{function: {name, ...}}, ...]` (or `[{name, ...}]`). Parsed via `json.loads()` with a try/except (malformed JSON → ignore that span's tools).
**Tool-call source** (`_llm_tool_calls(span)`):
- OTel: `gen_ai.output.messages[*].tool_calls[*].function.name`.
- neatlogs: each `neatlogs.llm.tool_calls.*` attribute is a JSON string of `{function: {name, ...}}`. Parsed with try/except.
**Computation:** `unused = sorted(all_defined - all_called)`. ONE finding per trace; evidence lists up to 3 example tool names.
**Related codes:** `("missing-span-kind",)`.
**Source:** `_token_waste_findings()` lines 1834–1855, helpers `_llm_tool_definitions()` lines 1667–1701 and `_llm_tool_calls()` lines 1704–1735.

#### C. Manual-fix snippets (`--emit-fix`)

The `--emit-fix <CODE>` CLI flag does NOT read the log file. It prints a copy-paste-able BEFORE/AFTER snippet for the given code (from `_FIX_SNIPPETS` dict) and exits 0. Unknown code → exit 2 with stderr message listing the known codes.

**The 4 registered snippets** (each is a `(description, before, after)` triple, lines 1931–1973):

| Code | Description | Before | After |
|---|---|---|---|
| `init-after-client` | Move `neatlogs.init()` to the top of the entry point. | `from openai import OpenAI; import neatlogs; neatlogs.init(api_key=...)` | `import neatlogs; neatlogs.init(api_key=...); from openai import OpenAI` |
| `missing-span-kind` | Add `kind='TOOL'` to the `@neatlogs.span` decorator. | `@trace\ndef my_function(): ...` | `@trace(kind='TOOL')\ndef my_function(): ...` |
| `zero-duration-span` | Wrap the body in `try: ... finally: span.end()`. | `span = tracer.start_span(...); response = orig(...); return response` | `span = tracer.start_span(...); try: return orig(...); finally: span.end()` |
| `error-status-no-event` | Call `record_exception(e)` inside the except block. | `except Exception as e: span.set_status(StatusCode.ERROR); raise` | `except Exception as e: span.set_status(StatusCode.ERROR, str(e)); span.record_exception(e); raise` |

**The render function** (`render_fix_snippet(code)`):
```python
def render_fix_snippet(code: str) -> str | None:
    if code not in _FIX_SNIPPETS:
        return None
    desc, before, after = _FIX_SNIPPETS[code]
    return (
        f"# Finding: {code}\n"
        f"# Suggested: {desc}\n\n"
        f"# BEFORE:\n{before}\n\n"
        f"# AFTER:\n{after}\n"
    )
```

**Why no AST-based auto-fix:** rewrites are fragile across project structures (Jupyter notebooks, K8s init containers, generated code). The user copy-pastes the snippet — correct-by-construction.

#### D. CLI flag wiring (new in PR #21)

| Flag | Effect |
|---|---|
| `--read-prompt-content` | Enables the `repeated-system-prompt` check. Default off (PII). |
| `--emit-fix CODE` | Print the manual-fix snippet for the given code. Exits 0 with the snippet on stdout, exits 2 on unknown code with stderr. Does NOT read the log file. |
| `path` (positional) | Now `nargs="?"` (optional). Required only when `--emit-fix` is not set. If both missing: `sys.exit(2)` with `error: a path is required (or use --emit-fix <code>)`. |

Source: `main()` lines 409–478.

#### E. Where the new checks run in the diagnose pipeline

The OTel GenAI and token-waste checks run alongside the other "always-run" pre-launch dimensions in `_diagnose_trace` (see §6.1 step 3). The new ordering is:

3. **Pre-launch reliability dimensions (always run, even on unusual traces):**
   - 3a. `_init_order_findings` → `init-after-client`
   - 3b. `_attribute_completeness_findings` → `missing-span-kind`
   - 3c. `_data_integrity_findings` → `zero-duration-span`, `error-status-no-event`, `latency-mismatch`
   - 3d. `_otel_genai_findings` → `otel-genai-missing`, `otel-genai-inconsistent` — NEW
   - 3e. `_token_waste_findings` → `oversized-prompt`, `repeated-system-prompt`, `unused-tool-definition` — NEW

These five dimensions run on EVERY trace shape. The OTel GenAI check reads `gen_ai.*` attrs and the token-waste check reads prompt sizes / tool names — both are tolerant of foreign-scope spans (they just don't fire on them unless the span is also an LLM kind).

---

## 5. Visibility rule

**Internal spans are excluded from most checks.** A span is internal if it has `neatlogs.internal=True` in attributes, OR if its name is exactly `neatlogs.trace.complete`. They are still counted in `spans_read` and `trace_count` but not in `visible`.

The visibility filter is applied BEFORE running checks. Use it as: `visible = [s for s in spans if not _is_internal(s)]`.

**Exception:** file-level counts (`spans_read`, `invalid_lines`) include internal spans. Run-level findings (`multi-run-log`, `pipeline-stage-summary`) operate on visible spans only.

---

## 6. Ordering and early-return — CRITICAL

The checks within `_diagnose_trace()` run in this specific order. Violating the order will produce false positives or miss findings:

### 6.1 Order
1. **Empty visible?** → return empty (no findings).
2. **Build `child_map`, `span_ids`, `duplicate_span_ids` (used by later checks).**
3. **Pre-launch reliability dimensions first** (always run, even on unusual traces):
   - 3a. `_init_order_findings` → `init-after-client`
   - 3b. `_attribute_completeness_findings` → `missing-span-kind`
   - 3c. `_data_integrity_findings` → `zero-duration-span`, `error-status-no-event`, `latency-mismatch`
   - 3d. `_otel_genai_findings` → `otel-genai-missing`, `otel-genai-inconsistent`
   - 3e. `_token_waste_findings` → `oversized-prompt`, `repeated-system-prompt`, `unused-tool-definition`
4. **`rootless-http-only` check** → if true, append finding and **return immediately** (only the 5 dimensions from step 3 have already run).
5. **`missing-root-kind` check** → append if no root in ROOT_KINDS.
6. **Hierarchy pathologies** (in this order):
   - `_orphan_parent_findings`
   - `_self_parent_findings`
   - `_duplicate_span_id_findings`
   - `_multiple_roots_findings`
   - `_cycle_findings`
7. **`_agent_without_llm_findings`** (subtree-based).
8. **`_missing_io_findings`**.

**Why this order matters:** the 5 pre-launch dimensions must run on EVERY trace shape. Pre-PR-#20, the `rootless-http-only` early-return caused data-integrity and init-after-client to be silently skipped on rootless HTTP traces. This was a real bug (fixed in `9669556`). The same reasoning extends to the new dimensions: OTel GenAI compliance and token-waste are useful diagnostic info even when the trace is structurally unusual.

**The `read_prompt_content` parameter** flows from `main()` → `diagnose()` → `_diagnose_trace()` → `_token_waste_findings(..., read_prompt_content=...)`. The kwarg is the only way the token-waste check knows whether to fire `repeated-system-prompt`. Default `False`.

### 6.2 Per-trace findings vs run-level findings
Per-trace findings carry `trace_id` and `run_id`. Run-level findings (`multi-run-log`, `pipeline-stage-summary`) carry `trace_id=None, run_id=None` in their finding — but the report's `run_count` and `trace_count` are computed from the full set.

### 6.3 Foreign-only filter
When `--foreign-only` is set, only `foreign-instrumentation-detected` findings are returned. **The `pipeline-stage-summary` is also suppressed** under this flag (foreign findings aren't the user's pipeline).

---

## 7. Cycle detection — detailed algorithm

Iterative DFS. Three sets per node: `done` (fully explored), `in_path` (currently on DFS stack). Walk DOWN via `child_map`. When a child id is in `in_path` (not `done`), it's a back-edge → cycle detected.

```
def find_cycles(visible_spans, child_map):
    done = set()
    in_path = set()
    cycles = []
    for root in roots:
        if root in done: continue
        # iterative DFS with explicit stack
        stack = [(root, iter(child_map.get(root, [])))]
        in_path.add(root)
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child in in_path:
                    cycles.append(back_edge)
                elif child not in done:
                    in_path.add(child)
                    stack.append((child, iter(child_map.get(child, []))))
            except StopIteration:
                stack.pop()
                in_path.discard(node)
                done.add(node)
    return cycles
```

**Filter self-cycles from the cycle walk** (they're reported as `self-parent` instead).

---

## 8. CLI behavior

`neatlogs-doctor [PATH] [--json] [--run-id ID] [--foreign-only] [--read-prompt-content] [--emit-fix CODE]`

| Flag | Effect |
|---|---|
| `PATH` | The span log file. Positional, optional (`nargs="?"`). Required only when `--emit-fix` is not set. If omitted AND `--emit-fix` is not set: `sys.exit(2)` with `error: a path is required (or use --emit-fix <code>)`. |
| `--json` | Output a JSON report instead of the human-readable report. |
| `--run-id ID` | Only analyze spans whose `session.id` matches. |
| `--foreign-only` | Only return `foreign-instrumentation-detected` findings. |
| `--read-prompt-content` | Enable the `repeated-system-prompt` check (PII-gated). Default off. |
| `--emit-fix CODE` | Print a manual-fix snippet for the given code; do NOT read the log file. Exits 0 with the snippet on stdout, exits 2 on unknown code with stderr message listing the known codes. |

**Exit codes:**
- 0 = success (no error-severity findings, or `--emit-fix` succeeded).
- 1 = at least one error-severity finding (only when analyzing a log file).
- 2 = argument error (path missing without `--emit-fix`, or `--emit-fix CODE` is unknown).

**Default output (human):** A human-readable report. See `format_report()` for the exact format — it includes a header, run/trace counts, and a sorted list of findings with severity icon, code, title, evidence, and suggestion.

**`--json` output:** A single JSON object with `path`, `spans_read`, `trace_count`, `run_count`, `invalid_lines`, `findings`. Each finding is the `to_dict()` form: `{severity, code, title, evidence, suggestion, trace_id?, run_id?, fix_class?, automated_fix_available?, doc_url?, related_codes?}`. The 6 new fields (`fix_class`, `automated_fix_available`, `doc_url`, `related_codes`) are only included when set (not None, not False, not empty tuple).

**Stable sort:** the findings list is sorted by `(severity_rank, code)`, where `severity_rank = {"error": 0, "warning": 1, "info": 2}`. Match this exactly for shell-script consumption.

---

## 9. LLM actionability surface

Four new fields on every finding (added in PR #20, all backward-compat default to None/False/()):

| Field | Type | Purpose |
|---|---|---|
| `fix_class` | `str \| None` | One of 8 values (see §3.2). LLM groups findings by class to decide what to fix first. |
| `automated_fix_available` | `bool` | True if a coding agent could fix without a human. Used to decide whether to attempt a fix. |
| `doc_url` | `str \| None` | A pointer to human-readable context. **MUST NOT be a 404.** See §11 for the in-repo path convention. |
| `related_codes` | `tuple[str, ...]` | Codes commonly seen together. LLM cross-references these. |

`to_dict()` only includes a field when it is set (not None, not False, not empty tuple). The default-empty behavior is the backward-compat path.

---

## 10. Test patterns — what to test

The Python test suite has 128 unit tests + 17 integration tests. Your TS/Go port should aim for parity. The minimum test categories:

### 10.1 Per-finding test (one happy + one negative per code)
For each of the **20** finding codes (14 from PR #20 + 6 from PR #21), write at least:
- A "fires when condition is met" test
- A "doesn't fire when condition isn't met" test (for the early-return / suppression cases)
- An "evidence contains the right example names" test
- For the OTel GenAI findings: a test that confirms `gen_ai.*` attrs are read (both formats) and a test that confirms foreign-scope/internal-span exclusion.
- For the token-waste findings: a test for the `--read-prompt-content=False` opt-out behavior.
- For the manual-fix snippets: a test that prints the snippet and asserts the BEFORE/AFTER format, plus a test for the unknown-code error path.

### 10.2 Framework coverage (51 tests)
17 frameworks × 3 checks = 51 parametrized tests. Frameworks: openai, anthropic, google_genai, vertex_ai, bedrock, cohere, mistral, groq, together, fireworks, langchain, llama_index, dspy, haystack, crewai, openai_agents, strands. Three checks per framework:
- Healthy trace → no findings. **The "healthy" fixture must include both neatlogs AND OTel GenAI attrs (gen_ai.operation.name, gen_ai.provider.name, gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens) on the LLM span. A trace that only has neatlogs attrs will trigger `otel-genai-missing` — that's the correct behavior.**
- Foreign-scope detection → `foreign-instrumentation-detected` fires.
- Missing-IO detection → `tool-missing-io` / `llm-missing-io` fires.

### 10.3 Performance tests
- 1K, 5K, 10K, 50K, 100K spans — must complete in <1s for 100K. Linear scaling.
- Cycle detection: same.

### 10.4 Edge cases
- Empty file → `no-spans` error.
- Non-UTF-8 file → graceful failure, `invalid-jsonl` finding.
- Internal spans → excluded from checks but counted in `spans_read`.
- Rootless HTTP-only trace → `rootless-http-only` fires; new dimensions still run.
- Multi-run log → `multi-run-log` fires.
- `--run-id` filter on a non-existent ID → `run-id-not-found` error.
- Diamond hierarchy (multiple paths to same grandchild) → no false-positive cycle.
- Self-parent + cycle co-occurring → both findings fire (cycle excludes self-cycles).

### 10.5 Backward compatibility
- Old `to_dict()` output (no new fields) — new fields absent.
- New fields when set — present.
- CLI: `main()` accepts the same args.
- Span log without `instrumentation_scope` — handled gracefully (warns via `scope-not-preserved`).

---

## 11. Doc URL convention — the bug we hit and the fix

**Pre-fix bug (PR #20 initial commit `45d5c50`):** The new dimensions' `doc_url` fields pointed to `https://docs.neatlogs.com/...` paths. **All three URLs returned 404.** The LLM actionability surface was a lie.

**Fix (commit `1dcd4d9`):** Point `doc_url` to **in-repo files** that actually exist. Use these paths:

| Finding code | doc_url |
|---|---|
| `init-after-client` | `skills/neatlogs/references/troubleshooting.md#1-import-order-issues-most-common-mistake` |
| `missing-span-kind` | `skills/neatlogs/references/troubleshooting.md#6-common-anti-patterns-table` |
| `tool-missing-io` / `llm-missing-io` / `retriever-missing-io` | (pre-existing — pre-fix these are 404 too; out of scope for the new dimensions but worth fixing in a follow-up) |

**Why in-repo paths:** the LLM agent that consumes the JSON report has the source tree. The anchor `#1-import-order-issues-most-common-mistake` points to the `troubleshooting.md` section 1 header slug. Verify the section headers in the source match your anchors (lowercase, hyphenated).

**Test:** `test_new_dimension_doc_urls_are_resolvable` in `test_doctor.py` asserts:
1. The `doc_url` does not contain `docs.neatlogs.com`.
2. The `doc_url` path resolves to an existing file relative to the repo root.

---

## 12. Bugs found and fixed during the 15-step review

These are bugs the Python implementation ALREADY had. Your TS/Go port must avoid them — or implement the fix from day one.

### 12.1 Span-leak in finalize phase (`init-after-client` missed on rootless HTTP)
**Symptom:** `rootless-http-only` early-return at the top of `_diagnose_trace()` caused the 4 new dimensions to be silently skipped on rootless HTTP traces.
**Fix (commit `9669556`):** Moved the new dimensions to run BEFORE the early-return.
**Test:** `test_diagnose_rootless_http_with_data_integrity_still_flags_both`.

### 12.2 Status format mismatch
**Symptom:** `error-status-no-event` only matched the Neatlogs-normalized format `{"code": "ERROR"}`. Foreign SDKs (or anyone running the doctor on a non-Neatlogs log) would emit `{"status_code": {"name": "ERROR", "value": 2}}` and the check would miss the error.
**Fix (commit `9669556`):** Added `_span_status_is_error()` helper that accepts BOTH formats. See §2.3.
**Test:** `test_data_integrity_error_no_event_sdk_status_format`.

### 12.3 doc_url points to 404
See §11.

### 12.4 Suggestion text references non-existent API
**Symptom:** `init-after-client` suggestion said "call neatlogs.init(force_reload=True)" but `neatlogs.init()` has no `force_reload` parameter.
**Fix (commit `e0b7106`):** Changed to "call neatlogs.shutdown() then neatlogs.init() again" — which is the actual escape hatch.
**Test:** `test_init_order_suggestion_does_not_reference_nonexistent_api`.

### 12.5 `pipeline-stage-summary.related_codes` hardcoded to init-stage classes
**Symptom:** The filter `f.fix_class in ("init_order", "config", "pipeline")` only catches init-stage classes. When the dominant stage was 'span' or 'instrument', `related_codes` was empty.
**Fix (commit `e0b7106`):** Extracted the `fix_class → stage` mapping into a module-level `_FIX_CLASS_TO_STAGE` dict; both the stage counter and the related_codes filter use it.
**Test:** `test_pipeline_stage_summary_related_codes_matches_dominant_stage`.

### 12.6 (Not a bug) `max(counts, key=counts.get)` is fragile
A consult-mmx review flagged this. It's not a bug — `dict.get(k, default=None)` works as a key function. But the more idiomatic `key=lambda k: counts[k]` is clearer. **Don't introduce the bug** by using `counts.get` and then changing the dict structure.

### 12.7 (Not a bug) Sub-microsecond zero-duration false positive
The consult flagged this as a potential issue but it's not: Neatlogs uses nanosecond-int timestamps, not floats. A sub-microsecond span would have `start == end == N` where N is some nanosecond value, and the check would fire correctly. **No fix needed.**

### 12.8 PR #21 review issues

**12.8.1 PII opt-in for `repeated-system-prompt` (the right call)**
A reviewer flagged that reading the system-prompt content of every LLM span is a PII leak. The implementation **correctly** defaults `read_prompt_content=False` and gates the `repeated-system-prompt` check on that flag. The other token-waste checks (`oversized-prompt` reads char counts, `unused-tool-definition` reads tool names) don't have PII concerns and run by default. This is the right design — the suggestion mentions `--read-prompt-content` in the help text and the `main()` docstring.

**12.8.2 OTel GenAI walk ordering (subtle but important)**
The `_otel_genai_findings` walker must check `_is_llm_kind(span)` BEFORE looking at `neatlogs.span.kind` and `gen_ai.operation.name` separately. If a span has `neatlogs.span.kind == "tool"` and `gen_ai.operation.name == "chat"`, the check must skip it (it's a tool span, not an LLM span, regardless of the OTel op-name). The test `test_otel_genai_does_not_fire_on_tool_span_with_chat_opname` verifies this.

**12.8.3 Fix-snippet source must be plain text, not JSON**
The snippets in `_FIX_SNIPPETS` are raw Python code with newlines. The renderer uses `\n` (Python's escape) to join lines, and the CLI writes the result to stdout verbatim. Don't JSON-encode the snippet — the user copy-pastes it into their editor. The test `test_emit_fix_output_is_plain_text` verifies the snippet is readable text, not a JSON-escaped string.

---

## 13. Implementation checklist (use this as your todo list)

- [ ] Define the input span-log schema (see §2). Tolerate missing keys with sensible defaults.
- [ ] Implement the status format tolerance (see §2.3).
- [ ] Implement the instrumentation_scope tolerance (string OR dict).
- [ ] Define the `DoctorFinding` / `DoctorReport` types with all 11 fields (severity, code, title, evidence, suggestion, trace_id, run_id, fix_class, automated_fix_available, doc_url, related_codes).
- [ ] Implement `to_dict()` with the "absent when unset" rule (NOT `null` for None fields).
- [ ] Implement run grouping (session.id → trace_id → sentinel).
- [ ] Implement visibility filter (exclude `neatlogs.internal` and `neatlogs.trace.complete`).
- [ ] Implement `_read_spans` (parse JSONL, track invalid lines).
- [ ] Implement the 12 trace-level checks in the exact order in §6.1.
- [ ] Implement the cycle detection algorithm in §7.
- [ ] Implement the missing-io checks (4 sub-types: llm/tool/retriever + the system-only LLM case).
- [ ] Implement the 5 pre-launch dimensions (init-order, attribute-completeness, data-integrity, OTel GenAI, token-waste).
- [ ] **NEW (PR #21):** Implement `_is_llm_kind(span)` — true if neatlogs `kind=llm` OR OTel `gen_ai.operation.name` ∈ {chat, text_completion, generate_content}.
- [ ] **NEW (PR #21):** Implement the 5 OTel GenAI attribute readers (operation_name, provider_name, request_model, usage_input_tokens, usage_output_tokens, response_finish_reasons) plus the input-messages / system-instructions / tool-definitions / output-messages walks for the token-waste check.
- [ ] **NEW (PR #21):** Implement `_llm_prompt_size(span)` — char count across OTel `gen_ai.input.messages` + neatlogs `neatlogs.llm.input_messages.*` + `neatlogs.llm.system` + `neatlogs.llm.prompts.*`.
- [ ] **NEW (PR #21):** Implement `_llm_system_prompt(span)` — prefer neatlogs `neatlogs.llm.system`, fall back to OTel `gen_ai.system_instructions[*].content` (string or list of `{type, text}` parts, joined with `\n`). Return None if neither.
- [ ] **NEW (PR #21):** Implement `_llm_tool_definitions(span)` and `_llm_tool_calls(span)` — set-returning helpers that walk OTel + neatlogs attr layouts. Parse JSON strings in neatlogs attrs with try/except.
- [ ] **NEW (PR #21):** Implement `OTEL_GENAI_LLM_OPERATIONS = {"chat", "text_completion", "generate_content"}`, `OVERSIZED_PROMPT_CHAR_THRESHOLD = 50_000`, `REPEATED_SYSTEM_PROMPT_THRESHOLD = 10`.
- [ ] **NEW (PR #21):** Implement the `_FIX_SNIPPETS` dict (4 entries) and `render_fix_snippet(code)` function.
- [ ] **NEW (PR #21):** Wire the `read_prompt_content` kwarg through `diagnose()` → `_diagnose_trace()` → `_token_waste_findings()`.
- [ ] Implement the file-level / run-level findings (multi-run-log, run-id-not-found, file-not-found, no-spans, invalid-jsonl, scope-not-preserved).
- [ ] Implement `format_report()` (human-readable) and `--json` (machine).
- [ ] Implement CLI flag parsing (--json, --run-id, --foreign-only, --read-prompt-content, --emit-fix CODE, optional path).
- [ ] Implement the exit code rule (0 = no error-severity findings, 1 otherwise, 2 = argument error including unknown --emit-fix code).
- [ ] Set `doc_url` to in-repo paths (see §11). **Do NOT use `https://docs.neatlogs.com/...` — that domain returns 404.**
- [ ] Set `automated_fix_available=True` for `init-after-client` only. All other findings default to False.
- [ ] Set the `related_codes` filter to use the same `_FIX_CLASS_TO_STAGE` dict as the stage counter (do NOT hardcode init-stage classes).
- [ ] Add tests covering all 20 finding codes (happy + negative per code).
- [ ] Add tests for the 5 new dimensions: `_is_llm_kind` recognizes both neatlogs kind and OTel op-name; `_otel_genai_findings` walks both attr layouts and skips tool spans with chat op-names; `_token_waste_findings` honors `read_prompt_content=False`; `_FIX_SNIPPETS` produces the right output for each of the 4 codes; unknown code returns None.
- [ ] Add 17 × 3 = 51 framework coverage tests. **The "healthy" fixture for each framework MUST include both neatlogs and OTel GenAI attrs (gen_ai.operation.name, gen_ai.provider.name, gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens) on the LLM span. A trace with only neatlogs attrs will fire `otel-genai-missing` — that's the correct behavior.**
- [ ] Add performance tests (1K, 5K, 10K, 50K, 100K spans — must be <1s for 100K).
- [ ] Add edge-case tests (empty, non-UTF-8, rootless HTTP, multi-run, etc.).
- [ ] Add backward-compat tests (to_dict() with and without new fields).

---

## 14. Parity verification

Once your TS/Go port is implemented, run it on the same test inputs as the Python reference and compare outputs:

1. **Byte-level diff** of `--json` output for the 3 E2E demo scripts in `docs/pr20-evidence/doctor_e2e_*.py` (these are also runnable as standalone Python scripts; rewrite the input generators in your target language and compare).

2. **Test-suite parity:** the 128 unit tests in `test_doctor.py` should port 1:1. Any deviation is a port bug.

3. **Performance parity:** 100K spans must complete in <1s on the same hardware. If it's slower, the cycle detection or `child_map` building is the likely culprit.

4. **Conformance with the E2E demo scripts** (each prints "ALL E2E TESTS PASSED" / "ALL DIMENSION TESTS PASSED" on success):
   - `docs/pr20-evidence/doctor_e2e_demo.py` — 5 bugs + 3 enhancements
   - `docs/pr20-evidence/doctor_e2e_dimensions.py` — 4 new dimensions + LLM actionability
   - `docs/pr20-evidence/doctor_e2e_realsdk.py` — Real SDK roundtrip

---

## 15. Performance — why the Python implementation is fast

For context, here are the key performance characteristics of the Python reference (Apple M-series, Python 3.11):

- **Cycle detection** is the most expensive check. The iterative DFS with 3 sets achieves O(V+E). Pre-fix (recursive with implicit `in_path`), it was O(V²) — 5K spans took 92 seconds; post-fix: 18ms.
- **`child_map` building** is O(N) and happens once per trace.
- **Group by run** is O(N) using a dict.
- **All checks** are O(N) or O(V+E) over visible spans.

For your TS/Go port, the same algorithmic complexity applies. Map/filter/iter are built-in. The hot loop is the cycle DFS — keep it iterative, not recursive (Python's stack overflows; Go/TS have stack limits too).

---

## 16. Reference: file map

| Path | Lines | Role |
|---|---|---|
| `neatlogs/doctor.py` | 2,000+ | The doctor. All top-level functions (33 in PR #20, +5 in PR #21 = 38). |
| `tests/unit/test_doctor.py` | 2,577+ | 142 unit tests (128 from PR #20 + 14 from PR #21). |
| `tests/integration/test_doctor_e2e.py` | 617 | 17 integration tests (CLI subprocess runs). |
| `docs/pr20-evidence/doctor_e2e_demo.py` | 21 KB | E2E demo for 5 bugs + 3 enhancements. |
| `docs/pr20-evidence/doctor_e2e_dimensions.py` | 11 KB | E2E demo for 4 new dimensions. |
| `docs/pr20-evidence/doctor_e2e_realsdk.py` | 3 KB | Real SDK roundtrip demo. |
| `docs/pr20-evidence/doctor_e2e_pr21.py` | 25 KB | E2E demo for PR #21 features (OTel GenAI, token-waste, --emit-fix) — 41 assertions. |
| `skills/neatlogs/references/troubleshooting.md` | — | The in-repo doc that `doc_url` fields point at. |
| `neatlogs/init.py` | 759 | The `neatlogs.init()` function. The `init-after-client` suggestion refers to its `shutdown()` companion. |
| `neatlogs/core/span_processor.py` | — | Where the SDK writes the span log files the doctor reads. Reference for the exact JSON shape. |

### Key constants added in PR #21

| Constant | Value | Where read |
|---|---|---|
| `OTEL_GENAI_OPERATION_NAME` | `"gen_ai.operation.name"` | `_is_llm_kind`, `_otel_genai_findings` |
| `OTEL_GENAI_PROVIDER_NAME` | `"gen_ai.provider.name"` | healthy-trace fixtures |
| `OTEL_GENAI_REQUEST_MODEL` | `"gen_ai.request.model"` | healthy-trace fixtures |
| `OTEL_GENAI_USAGE_INPUT_TOKENS` | `"gen_ai.usage.input_tokens"` | healthy-trace fixtures |
| `OTEL_GENAI_USAGE_OUTPUT_TOKENS` | `"gen_ai.usage.output_tokens"` | healthy-trace fixtures |
| `OTEL_GENAI_RESPONSE_FINISH_REASONS` | `"gen_ai.response.finish_reasons"` | healthy-trace fixtures |
| `OTEL_GENAI_LLM_OPERATIONS` | `frozenset({"chat", "text_completion", "generate_content"})` | `_is_llm_kind`, `_otel_genai_findings` |
| `OVERSIZED_PROMPT_CHAR_THRESHOLD` | `50_000` | `_token_waste_findings` |
| `REPEATED_SYSTEM_PROMPT_THRESHOLD` | `10` | `_token_waste_findings` |
| `_FIX_SNIPPETS` | dict of 4 `(description, before, after)` triples | `render_fix_snippet` |

### Key helpers added in PR #21

| Function | Returns | Purpose |
|---|---|---|
| `_is_llm_kind(span)` | `bool` | True if neatlogs kind=llm OR OTel op in `OTEL_GENAI_LLM_OPERATIONS` |
| `_otel_genai_findings(spans, trace_id, run_id)` | `list[DoctorFinding]` | Returns 0–2 findings (otel-genai-missing, otel-genai-inconsistent) |
| `_llm_prompt_size(span)` | `int` | Char count across OTel + neatlogs attrs |
| `_llm_system_prompt(span)` | `str \| None` | System prompt text or None |
| `_llm_tool_definitions(span)` | `set[str]` | Tool names defined |
| `_llm_tool_calls(span)` | `set[str]` | Tool names called |
| `_token_waste_findings(spans, trace_id, run_id, *, read_prompt_content)` | `list[DoctorFinding]` | Returns 0–3 findings |
| `render_fix_snippet(code)` | `str \| None` | Plain-text snippet or None for unknown code |

---

## 17. Quick-start for the TS/Go implementer

```bash
# 1. Clone the reference and check out the doctor branch
git clone https://github.com/Harsh23Kashyap/neatlogs.git
cd neatlogs
git checkout feat/trace-doctor  # @ e0b7106

# 2. Run the existing tests so you know what "passing" means
/opt/homebrew/bin/python3.11 -m pytest tests/unit/test_doctor.py tests/integration/test_doctor_e2e.py

# 3. Run the E2E demos to see expected output formats
/opt/homebrew/bin/python3.11 docs/pr20-evidence/doctor_e2e_demo.py
/opt/homebrew/bin/python3.11 docs/pr20-evidence/doctor_e2e_dimensions.py
/opt/homebrew/bin/python3.11 docs/pr20-evidence/doctor_e2e_realsdk.py

# 4. Now read the test file: tests/unit/test_doctor.py
#    Each test is a spec for one finding. Translate test → implementation in your target language.

# 5. Read neatlogs/doctor.py section by section. Cross-reference every test.
```

---

## 18. What didn't work (avoid these dead ends)

- **Recursive cycle detection.** Pre-fix, used implicit call stack. Hit Python's recursion limit at ~1K spans. Iterative DFS with explicit stack is the fix.
- **Hardcoded `related_codes` filter (`init_order`, `config`, `pipeline`).** Worked for init-stage dominance but produced empty `related_codes` for span/instrument/hierarchy dominance. Use the same `_FIX_CLASS_TO_STAGE` dict for both counter and filter.
- **Pointing `doc_url` to `docs.neatlogs.com`.** Domain returns 404. Use in-repo paths with anchors into the troubleshooting file.
- **Referencing `force_reload=True` in the init suggestion.** The kwarg doesn't exist on `neatlogs.init()`. Use `neatlogs.shutdown()` + `neatlogs.init()`.
- **Filtering internal spans from data-integrity checks.** Don't — the user might want to see them. But also don't double-count them in `spans_read`. Use the `visible` filter only for per-trace checks; count all spans in the file-level totals.
- **Self-parent as a cycle.** A span pointing to itself is a self-cycle, but reporting it as both `self-parent` and `cycle` is a double-report. Filter self-cycles out of the cycle walk; report them via `_self_parent_findings` only.

---

## 19. Open questions for the port implementer

- **Q1: Performance vs. Python.** TS is typically 2-5x faster than Python for this kind of work; Go is 10-20x faster. The cycle-detection iterative DFS is the hot path. If your port is slower than the Python reference, profile there.
- **Q2: JSON parser.** Use a streaming JSONL parser. Don't read the whole file into memory before parsing. The Python reference uses `for line in f: span = json.loads(line)` which is streaming. Your port should match.
- **Q3: Stdout for `--json`.** Write the JSON object to stdout, not stderr. Diagnostic messages go to stderr. This is shell-idiomatic.
- **Q4: Exit code.** 0 if no error-severity findings, 1 otherwise. The reference uses this so `set -e` and CI gating work.
- **Q5: Stable output.** The findings list is sorted by (severity, code) — `(0=error, 1=warning, 2=info)`, then alphabetical by code. Match this exactly for shell-script consumption.
- **Q6: No network.** The doctor is offline. Don't add any auto-update check, telemetry, or remote fetch. The whole point is local-only debugging.

---

## 20. Summary of acceptance criteria

A TS/Go port is "done" when:

1. **All 20 finding codes** (the 14 pre-PR-#21 + 6 from PR #21) fire correctly across the existing test suite.
2. **All 6 bugs** listed in §12 are avoided (the port implements the fixes from day one).
3. **All 51 framework coverage tests** (17 × 3) pass, with the "healthy" fixture including both neatlogs and OTel GenAI attrs.
4. **Performance:** 100K spans in <1s on comparable hardware.
5. **CLI:** `--json`, `--run-id`, `--foreign-only`, `--read-prompt-content`, `--emit-fix CODE`, optional path, exit codes 0/1/2.
6. **Backward compat:** the JSON report for an old log file (no `instrumentation_scope`, no new fields on findings) is the same shape as the Python reference's output for the same input.
7. **Doc URLs:** point to in-repo paths, never to `docs.neatlogs.com`.
8. **Suggestion text:** the `init-after-client` suggestion does not mention `force_reload=True`.
9. **PII opt-in:** `repeated-system-prompt` only fires when `read_prompt_content=True`; the other token-waste checks run by default.
10. **Manual-fix snippets:** `--emit-fix <code>` exits 0 with the snippet on stdout, exits 2 on unknown code with stderr listing the known codes. The snippet is plain text, not JSON-escaped.

When you can pass all 10 criteria, the port is feature-complete and you can ship.

---

# Language-specific port notes

The previous sections are language-agnostic. This section gives concrete TS/Go guidance: type signatures, idiomatic patterns, package recommendations, and port-specific gotchas. Read your language's section, then the language-agnostic sections above.

---

## 21. TypeScript port

### 21.1 Target

- **Node.js 18+** (LTS at time of writing). Built-in `fetch`, `node:test`, `--watch`, ESM. No transpiler needed.
- **Package manager:** pnpm or npm. Pin Node 18+ in `engines.node`.
- **Module system:** ESM (`"type": "module"` in `package.json`). Don't use CJS — the dynamic-import and worker-thread story is much cleaner in ESM.
- **TypeScript:** 5.4+ for `const` type parameters and improved narrowing.
- **Strict mode:** `strict: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`. The `noUncheckedIndexedAccess` is critical — it forces you to handle `undefined` on every array/dict access, which catches most of the bugs the Python reference found during review.
- **Runtime dependencies:** none. `process`, `fs`, `readline`, `path`, `node:test` are all built-in.
- **Dev dependencies:** `typescript`, `@types/node`, optionally `vitest` (faster than `node:test` for parametrized tests).

### 21.2 Type definitions

```typescript
// Span log schema — every field is optional except trace_id, span_id, name, kind.
type SpanKind = "workflow" | "chain" | "agent" | "tool" | "llm" |
                "embedding" | "reranker" | "retriever" | "http" | "mcp_tool" |
                string; // unknown kinds are tolerated

type InstrumentationScope = string | { name: string; version?: string };

type SpanStatus = {
  code?: string;            // "OK" | "ERROR" | "ERROR_STATUS" | "UNSET"
  description?: string;
  status_code?: string | { name?: string; value?: number };
};

type Span = {
  trace_id: string;
  span_id: string;
  parent_span_id?: string | null;
  name: string;
  kind: SpanKind;
  start_time?: number;      // ns epoch
  end_time?: number;        // ns epoch
  duration_ns?: number;
  status?: SpanStatus;
  events?: Array<{ name: string; attributes?: Record<string, unknown> }>;
  attributes?: Record<string, unknown>;
  instrumentation_scope?: InstrumentationScope;
};

type Severity = "error" | "warning" | "info";

type FixClass =
  | "init_order" | "attribute" | "capture" | "config"
  | "pipeline" | "hierarchy" | "instrumentation" | "data_integrity"
  | "none";

// Finding is a discriminated union by code — this lets TypeScript narrow
// type-safely when handling specific codes. For most consumers, the simple
// shape below is enough.
type Finding = {
  severity: Severity;
  code: string;
  title: string;
  evidence: string;
  suggestion: string;
  trace_id?: string;
  run_id?: string;
  fix_class?: FixClass;
  automated_fix_available?: boolean;
  doc_url?: string;
  related_codes?: readonly string[];
};

type Report = {
  path: string;
  spans_read: number;
  trace_count: number;
  run_count: number;
  invalid_lines: number[];
  findings: readonly Finding[];
};
```

**Why `readonly`:** the Python reference uses `tuple` and `@dataclass(frozen=True)`. The TS port should be `readonly` arrays and `as const` literals so the report can be hashed and diffed in tests.

**Why `FixClass` is a union of literals, not `string`:** TypeScript will catch typos at the call site (e.g. `fix_class: "data_intergity"` is a compile error).

### 21.3 Idiomatic patterns

**Streaming JSONL with `readline`:**
```typescript
import { createInterface } from "node:readline";
import { createReadStream } from "node:fs";

async function* readJsonl(path: string): AsyncGenerator<Span> {
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      yield JSON.parse(line) as Span;
    } catch {
      // Track the line number in the caller.
      throw new Error(`Invalid JSON at line ${rl.line - 1}`);
    }
  }
}
```

**Set operations for foreign-instrumentation and tool names:**
```typescript
const foreign = new Set<string>();
for (const span of visible) {
  const name = scopeName(span.instrumentation_scope);
  if (name && !name.startsWith("neatlogs.")) foreign.add(name);
}
```

**Tuple as `readonly` array (with length constraint):**
```typescript
const relatedCodes = ["otel-genai-missing"] as const;
// type: readonly ["otel-genai-missing"]
```

**Default-empty / absent-when-unset (matches Python's `to_dict` "absent when unset" rule):**
```typescript
function findingToDict(f: Finding): Record<string, unknown> {
  const out: Record<string, unknown> = {
    severity: f.severity,
    code: f.code,
    title: f.title,
    evidence: f.evidence,
    suggestion: f.suggestion,
  };
  if (f.trace_id) out.trace_id = f.trace_id;
  if (f.run_id) out.run_id = f.run_id;
  if (f.fix_class) out.fix_class = f.fix_class;
  if (f.automated_fix_available) out.automated_fix_available = f.automated_fix_available;
  if (f.doc_url) out.doc_url = f.doc_url;
  if (f.related_codes && f.related_codes.length) {
    out.related_codes = [...f.related_codes];
  }
  return out;
}
```

**Stable sort by severity then code:**
```typescript
const severityRank = { error: 0, warning: 1, info: 2 } as const;
findings.sort((a, b) => {
  const s = severityRank[a.severity] - severityRank[b.severity];
  return s !== 0 ? s : a.code.localeCompare(b.code);
});
```

### 21.4 CLI

Use Node's built-in `node:util.parseArgs` (Node 18.3+). It's the closest equivalent to Python's `argparse` and doesn't need a third-party dep.

```typescript
import { parseArgs } from "node:util";

const args = parseArgs({
  options: {
    json: { type: "boolean" },
    "run-id": { type: "string" },
    "foreign-only": { type: "boolean" },
    "read-prompt-content": { type: "boolean" },
    "emit-fix": { type: "string" },
  },
  allowPositionals: true,
  strict: true,
});

const path = args.positionals[0];
if (args.values["emit-fix"] !== undefined) {
  const snippet = renderFixSnippet(args.values["emit-fix"]);
  if (snippet === null) {
    process.stderr.write(
      `Unknown finding code: '${args.values["emit-fix"]}'. Known: ${Object.keys(FIX_SNIPPETS).sort().join(", ")}\n`,
    );
    process.exit(2);
  }
  process.stdout.write(snippet);
  process.exit(0);
}

if (path === undefined) {
  process.stderr.write("error: a path is required (or use --emit-fix <code>)\n");
  process.exit(2);
}
```

**Don't use `commander` or `yargs` for this.** They're heavier and have different exit-code conventions. `parseArgs` is enough for the 5 flags we have.

### 21.5 Output streams

- `process.stdout.write(...)` for the report and `--emit-fix` snippet.
- `process.stderr.write(...)` for errors and the unknown-code message.
- `process.exit(0 | 1 | 2)` for exit codes. Match the Python reference exactly: 0 = no error findings or `--emit-fix` success, 1 = error findings, 2 = argument error.
- **Don't use `console.log`** for the report — it appends a newline and writes to stdout, but `process.stdout.write` is more explicit and matches the Python `sys.stdout.write` behavior.

### 21.6 Testing

- Use `node:test` (built-in) for the 142 unit tests. It's fast and has no deps.
- Use `node:test`'s `describe` / `it` / `before` / `after` blocks. The syntax matches `vitest` closely.
- For parametrized tests (the 51 framework coverage tests), use `it.each(FRAMEWORKS)("healthy trace for %s has no findings", ...)` — `node:test` supports this since 22.x. On Node 18, use a `for` loop with `it()` calls inside.
- For E2E CLI tests, spawn a child process via `node:child_process.spawn` and capture stdout / stderr / exit code.
- Snapshot the `--json` output and diff against a fixture. The Python reference's `doctor_e2e_pr21.py` output (`docs/pr20-evidence/doctor_e2e_pr21_output.txt`) is the gold standard.

### 21.7 Lint and format

- **ESLint:** `eslint:recommended` + `@typescript-eslint/recommended` + `@typescript-eslint/recommended-require-type-checking`. Enable `no-floating-promises` (catches missing `await`).
- **Prettier:** default config. The Python repo doesn't use prettier, but the TS port should — `formatReport` has long strings that benefit from consistent line wrapping.
- **Type check:** `tsc --noEmit` in CI. Should pass with 0 errors.

### 21.8 TS-specific gotchas

1. **JSON numbers are doubles.** `gen_ai.usage.input_tokens` is an int in OTel but a `number` in JSON, which is a double in JS. For counts under 2^53, this is fine. For trace IDs (`uint128`), use `string` — don't try to convert to BigInt.
2. **Dictionary key access is `T | undefined` with `noUncheckedIndexedAccess`.** The cycle-detection algorithm uses `Map<string, Span[]>` — get returns `Span[] | undefined`. Always check for undefined.
3. **`process.argv[0]` is `node`, `[1]` is the script path, `[2:]` are user args.** `parseArgs` already handles this. Don't write your own.
4. **`JSON.parse` on the JSONL line is synchronous.** If the file is large (1M+ lines), this becomes the bottleneck. Consider a worker thread for the parse pass. The Python reference uses synchronous `json.loads` per line and is fast enough; TS `JSON.parse` is ~5x faster than Python's, so the same algorithm will be faster.
5. **ESM path imports need `.js` extensions** even for `.ts` source. Use `tsc` with `"moduleResolution": "NodeNext"` and `"module": "NodeNext"`. Or use a bundler (esbuild) for the CLI binary.
6. **`as const` is your friend for finding-code lists.** `const CODES = ["init-after-client", "missing-span-kind", ...] as const;` gives you a readonly tuple of literal types.
7. **The `to_dict` "absent when unset" rule is important.** TypeScript's `JSON.stringify` will serialize `undefined` values as absent — but if you use a class instance, it serializes as `null` for nullable fields. The TypeScript port must use plain object literals, not class instances, to match the Python behavior.

### 21.9 Quick-start

```bash
mkdir neatlogs-doctor-ts && cd neatlogs-doctor-ts
npm init -y
# edit package.json: "type": "module", "engines": { "node": ">=18.3" }
npm install --save-dev typescript @types/node
# create tsconfig.json (strict, NodeNext, ES2022)
# copy src/doctor.ts from your port
# copy tests/doctor.test.ts from your port
npx tsc --noEmit              # type check
node --test                   # run unit tests
node dist/cli.js ../spans.log # run the CLI
```

---

## 22. Go port

### 22.1 Target

- **Go 1.22+** for `slices`, `maps`, `cmp.Or`, and improved range over ints. The `iter` package (1.23) is nice for the cycle-detection generator, but not required.
- **Module path:** `github.com/neatlogs/neatlogs-doctor` or your org's path.
- **No third-party dependencies for the core.** The `flag` package (built-in) handles CLI args. `bufio` for streaming. `encoding/json` for parsing. `testing` for tests.
- **Optional dev deps:** `github.com/stretchr/testify` for assertion ergonomics, `golangci-lint` for lint.

### 22.2 Type definitions

```go
package doctor

type SpanKind string

const (
    KindWorkflow SpanKind = "workflow"
    KindChain    SpanKind = "chain"
    KindAgent    SpanKind = "agent"
    KindTool     SpanKind = "tool"
    KindLLM      SpanKind = "llm"
    // ... and IO_KINDS, ROOT_KINDS as maps
)

// InstrumentationScope — Go doesn't have union types, so we use an interface
// or a struct with a "kind" discriminator. The struct-with-pointer approach
// is cleaner here.
type InstrumentationScope struct {
    Name    string `json:"name,omitempty"`
    Version string `json:"version,omitempty"`
}

// UnmarshalJSON handles BOTH forms: "neatlogs.core.context" (string) and
// {"name": "neatlogs.core.context", "version": "1.4.20"} (object).
func (s *InstrumentationScope) UnmarshalJSON(data []byte) error {
    var str string
    if err := json.Unmarshal(data, &str); err == nil {
        s.Name = str
        return nil
    }
    type alias InstrumentationScope
    var obj alias
    if err := json.Unmarshal(data, &obj); err != nil {
        return err
    }
    *s = InstrumentationScope(obj)
    return nil
}

type SpanStatus struct {
    Code        string          `json:"code,omitempty"`
    Description string          `json:"description,omitempty"`
    StatusCode  json.RawMessage `json:"status_code,omitempty"`
}

type Span struct {
    TraceID             string               `json:"trace_id"`
    SpanID              string               `json:"span_id"`
    ParentSpanID        *string              `json:"parent_span_id,omitempty"`
    Name                string               `json:"name"`
    Kind                SpanKind             `json:"kind"`
    StartTime           *int64               `json:"start_time,omitempty"`
    EndTime             *int64               `json:"end_time,omitempty"`
    DurationNS          *int64               `json:"duration_ns,omitempty"`
    Status              *SpanStatus          `json:"status,omitempty"`
    Events              []SpanEvent          `json:"events,omitempty"`
    Attributes          map[string]any       `json:"attributes,omitempty"`
    InstrumentationScope *InstrumentationScope `json:"instrumentation_scope,omitempty"`
}

type Severity string

const (
    SeverityError   Severity = "error"
    SeverityWarning Severity = "warning"
    SeverityInfo    Severity = "info"
)

type FixClass string

const (
    FixInitOrder      FixClass = "init_order"
    FixAttribute      FixClass = "attribute"
    FixCapture        FixClass = "capture"
    FixConfig         FixClass = "config"
    FixPipeline       FixClass = "pipeline"
    FixHierarchy      FixClass = "hierarchy"
    FixInstrumentation FixClass = "instrumentation"
    FixDataIntegrity  FixClass = "data_integrity"
)

type Finding struct {
    Severity              Severity   `json:"severity"`
    Code                  string     `json:"code"`
    Title                 string     `json:"title"`
    Evidence              string     `json:"evidence"`
    Suggestion            string     `json:"suggestion"`
    TraceID               *string    `json:"trace_id,omitempty"`
    RunID                 *string    `json:"run_id,omitempty"`
    FixClass              *FixClass  `json:"fix_class,omitempty"`
    AutomatedFixAvailable bool       `json:"automated_fix_available,omitempty"`
    DocURL                *string    `json:"doc_url,omitempty"`
    RelatedCodes          []string   `json:"related_codes,omitempty"`
}

type Report struct {
    Path         string    `json:"path"`
    SpansRead    int       `json:"spans_read"`
    TraceCount   int       `json:"trace_count"`
    RunCount     int       `json:"run_count"`
    InvalidLines []int     `json:"invalid_lines"`
    Findings     []Finding `json:"findings"`
}

func (r *Report) HasErrors() bool {
    for _, f := range r.Findings {
        if f.Severity == SeverityError {
            return true
        }
    }
    return false
}
```

**Key Go-specific decisions:**

- **Pointers for optional fields.** `*string`, `*int64`, `*SpanStatus`. The `omitempty` JSON tag drops the field from the output when nil — matching the Python "absent when unset" rule.
- **`json.RawMessage` for `StatusCode`** because it can be either a string (`"ERROR"`) or an object (`{"name": "ERROR", "value": 2}`). Use a custom unmarshaler to handle both.
- **`map[T]struct{}` for sets.** Go has no built-in set type. `map[string]struct{}{}` is the idiomatic equivalent of Python's `set()` or TS's `Set<string>`.
- **No `tuple`.** Use a `[]string` for `RelatedCodes` and a comment noting that order is not significant but the field is a slice for JSON-array compatibility.

### 22.3 Idiomatic patterns

**Streaming JSONL with `bufio.Scanner`:**
```go
func readSpans(path string) (spans []Span, invalidLines []int, err error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, nil, err
    }
    defer f.Close()

    scanner := bufio.NewScanner(f)
    // Span lines can be large; bump the buffer to 10MB to handle big spans.
    scanner.Buffer(make([]byte, 64*1024), 10*1024*1024)
    lineNum := 0
    for scanner.Scan() {
        lineNum++
        line := bytes.TrimSpace(scanner.Bytes())
        if len(line) == 0 {
            continue
        }
        var span Span
        if err := json.Unmarshal(line, &span); err != nil {
            invalidLines = append(invalidLines, lineNum)
            continue
        }
        spans = append(spans, span)
    }
    return spans, invalidLines, scanner.Err()
}
```

**Cycle detection (iterative DFS, ported from §7):**
```go
func findCycles(visible []Span, childMap map[string][]string) []CycleEdge {
    done := make(map[string]bool)
    inPath := make(map[string]bool)
    var cycles []CycleEdge

    // Build a set of visible span IDs.
    visibleIDs := make(map[string]bool)
    for _, s := range visible {
        if s.SpanID != "" {
            visibleIDs[s.SpanID] = true
        }
    }

    for _, s := range visible {
        if !visibleIDs[s.SpanID] || done[s.SpanID] {
            continue
        }
        // Iterative DFS: stack frames are (node, children-iter).
        type frame struct {
            node     string
            children []string
            idx      int
        }
        stack := []frame{{node: s.SpanID, children: childMap[s.SpanID]}}
        inPath[s.SpanID] = true
        for len(stack) > 0 {
            top := &stack[len(stack)-1]
            if top.idx >= len(top.children) {
                // Done with this node.
                inPath[top.node] = false
                done[top.node] = true
                stack = stack[:len(stack)-1]
                continue
            }
            child := top.children[top.idx]
            top.idx++
            if inPath[child] {
                if child != top.node { // self-parent handled separately
                    cycles = append(cycles, CycleEdge{From: top.node, To: child})
                }
                continue
            }
            if done[child] {
                continue
            }
            inPath[child] = true
            stack = append(stack, frame{node: child, children: childMap[child]})
        }
    }
    return cycles
}
```

**Iterative DFS is critical.** Go's goroutine stack is dynamic, so a recursive DFS wouldn't hit a stack limit per se, but the iterative version is 2-3x faster and matches the Python reference's optimization.

**Set operations:**
```go
// All foreign scope names (set semantics via map[T]struct{}).
foreign := make(map[string]struct{})
for _, s := range visible {
    if s.InstrumentationScope == nil {
        continue
    }
    name := s.InstrumentationScope.Name
    if !strings.HasPrefix(name, "neatlogs.") {
        foreign[name] = struct{}{}
    }
}
```

**Stable sort:**
```go
severityRank := map[Severity]int{
    SeverityError:   0,
    SeverityWarning: 1,
    SeverityInfo:    2,
}
sort.SliceStable(findings, func(i, j int) bool {
    if severityRank[findings[i].Severity] != severityRank[findings[j].Severity] {
        return severityRank[findings[i].Severity] < severityRank[findings[j].Severity]
    }
    return findings[i].Code < findings[j].Code
})
```

**The `_FIX_CLASS_TO_STAGE` dict:**
```go
var fixClassToStage = map[FixClass]string{
    FixInitOrder:       "init",
    FixConfig:          "init",
    FixPipeline:        "init",
    FixInstrumentation: "instrument",
    FixCapture:         "instrument",
    FixDataIntegrity:   "span",
    FixAttribute:       "span",
    FixHierarchy:       "hierarchy",
}
```

### 22.4 CLI

Go's `flag` package is enough. Subcommands aren't needed — flags are flat.

```go
func main() {
    var (
        jsonOut            = flag.Bool("json", false, "Output JSON report.")
        runID              = flag.String("run-id", "", "Only analyze this run.")
        foreignOnly        = flag.Bool("foreign-only", false, "Only foreign-instrumentation findings.")
        readPromptContent  = flag.Bool("read-prompt-content", false, "Read LLM prompts (PII).")
        emitFix            = flag.String("emit-fix", "", "Print a fix snippet for the given code and exit.")
    )
    flag.Parse()
    args := flag.Args()

    if *emitFix != "" {
        snippet, ok := renderFixSnippet(*emitFix)
        if !ok {
            fmt.Fprintf(os.Stderr,
                "Unknown finding code: %q. Known: %s\n",
                *emitFix, strings.Join(sortedKeys(fixSnippets), ", "),
            )
            os.Exit(2)
        }
        fmt.Fprint(os.Stdout, snippet)
        os.Exit(0)
    }

    if len(args) == 0 {
        fmt.Fprintln(os.Stderr, "error: a path is required (or use --emit-fix <code>)")
        os.Exit(2)
    }

    report := Diagnose(args[0], DiagnoseOptions{
        RunID:             *runID,
        ForeignOnly:       *foreignOnly,
        ReadPromptContent: *readPromptContent,
    })
    if *jsonOut {
        enc := json.NewEncoder(os.Stdout)
        enc.SetIndent("", "  ")
        enc.Encode(report.ToDict())
    } else {
        fmt.Fprintln(os.Stdout, FormatReport(report))
    }
    if report.HasErrors() {
        os.Exit(1)
    }
}
```

**`go-playground/validator` is overkill.** Plain `flag` handles the 5 flags fine.

### 22.5 Output streams

- `fmt.Fprintln(os.Stdout, ...)` for the report and `--emit-fix` snippet.
- `fmt.Fprintln(os.Stderr, ...)` for errors and the unknown-code message.
- `os.Exit(0 | 1 | 2)`. Match the Python reference exactly.

### 22.6 Testing

- Use the built-in `testing` package. `go test ./...` is fast and has no deps.
- For parametrized tests (the 51 framework coverage tests), use `t.Run(name, func(t *testing.T) { ... })` inside a `for` loop. The Go 1.22+ `testing.B.Loop` for benchmarks is also worth using.
- For E2E CLI tests, use `os/exec` to run the built binary and capture stdout / stderr / exit code. Compare against a fixture file via `testdata/`.
- For snapshot tests, `github.com/stretchr/testify/assert` + `assert.Equal(t, expected, actual)` is the standard. If you don't want testify, write a small `diff` helper using `reflect.DeepEqual`.
- Coverage: `go test -cover ./...` should report 95%+ on `doctor/`.

### 22.7 Lint and format

- `gofmt` (built-in) — must pass.
- `go vet` (built-in) — must pass.
- `golangci-lint run` with the default config — must pass. The `govet`, `staticcheck`, `unused`, and `errcheck` linters are the most useful.
- `gocritic` is optional but useful for catching `if x, _ := y; ...` patterns.

### 22.8 Go-specific gotchas

1. **No `frozenset`.** Use `map[T]struct{}` for set semantics. The empty struct (`struct{}`) takes zero bytes.
2. **No `tuple`.** Use a struct with positional fields, or a `[]T` with a comment. For the 4 `(description, before, after)` fix snippets, a struct is cleaner:
   ```go
   type fixSnippet struct {
       description string
       before      string
       after       string
   }
   var fixSnippets = map[string]fixSnippet{
       "init-after-client": {
           description: "Move neatlogs.init() to the top of the entry point (before any LLM client is constructed).",
           before:      "from openai import OpenAI\nimport neatlogs\nneatlogs.init(api_key=os.environ['NEATLOGS_API_KEY'])\n",
           after:       "import neatlogs\nneatlogs.init(api_key=os.environ['NEATLOGS_API_KEY'])\nfrom openai import OpenAI\n",
       },
       // ...
   }
   ```
3. **Pointers for optional fields.** `*string`, `*int64`, etc. The `omitempty` JSON tag drops the field when nil. The Python reference's `to_dict` rule maps directly to this.
4. **Errors are values, not exceptions.** Every function that can fail returns `error`. Wrap with `fmt.Errorf("...: %w", err)` for context.
5. **JSON unmarshal is strict by default.** If the input has extra fields, they are silently dropped. If a field is `int` and the input has a string, `Unmarshal` returns an error. The doctor must be tolerant: use `map[string]any` for `Attributes` and use custom unmarshalers for the union types (Status, InstrumentationScope).
6. **`bufio.Scanner` has a default 64KB line limit.** Span lines can be large (50KB+ for a long prompt). Bump the buffer to 10MB. The Python reference doesn't have this problem because Python's `for line in f` is unbounded.
7. **`encoding/json` is slow for many small objects.** The doctor processes 100K+ spans in some cases. Consider `jsoniter` (drop-in replacement, 3-5x faster) for hot loops, or use `goccy/go-json` (5-10x faster). Both have compatible APIs.
8. **No `Optional<T>`.** Use a pointer `*T` and check for nil. For `omitempty` to work in JSON, the field must be a pointer or have a zero-value check.
9. **Map iteration order is randomized.** When you build a `map[string]int` and iterate it to compute the dominant stage, the order is non-deterministic. The Python reference uses `max(counts, key=counts.get)` which is also non-deterministic. The Go port should match: don't sort the stage list for the `pipeline-stage-summary` evidence; just emit them in iteration order. The test for that finding is fuzzy (it checks the count breakdown, not the order).
10. **String concatenation in the format_report function.** The Python reference uses `"\n".join(lines)`. The Go port should use a `strings.Builder` to avoid quadratic-time concatenation:
    ```go
    var b strings.Builder
    b.WriteString("Trace Doctor\n")
    b.WriteString(fmt.Sprintf("File: %s\n", report.Path))
    // ...
    return b.String()
    ```

### 22.9 Quick-start

```bash
mkdir neatlogs-doctor-go && cd neatlogs-doctor-go
go mod init github.com/neatlogs/neatlogs-doctor
# create doctor/doctor.go with your port
# create doctor/doctor_test.go with the 142 unit tests
go test ./...              # run unit tests
go vet ./...               # static analysis
go build -o bin/doctor ./cmd/doctor
./bin/doctor ../spans.log  # run the CLI
```

---

## 23. Side-by-side: TS vs Go tradeoffs

For the team picking which language to port first, here's the trade-off:

| Dimension | TypeScript | Go |
|---|---|---|
| **Lines of code** | ~2,500 (logic) + 800 (types) = 3,300 | ~2,200 (logic + types) |
| **Time to first working port** | 1-2 days (parse + cycle detect + 1 check) | 1-2 days (same) |
| **Performance vs Python** | 2-5x faster | 10-20x faster |
| **Distribution** | Single binary via `pkg` or `bun build`; or run with `node`. 30-50MB. | Single static binary via `go build`. 5-10MB. |
| **Stdlib coverage** | Built-in test runner, parseArgs, readline. Needs `node:test` 22+ for parametrized. | Built-in everything. Test runner is mature. |
| **Ecosystem for OTel** | `@opentelemetry/api` is the reference implementation. TS users are likely already OTel users. | `go.opentelemetry.io/otel` is the reference. Go OTel SDK has a different surface (e.g. `attribute.String` is a typed key-value). |
| **Type safety for finding codes** | String-literal union + `as const` for finding lists. | Typed `const` blocks + `map[Code]FindingHandler`. |
| **Easiest first port target** | Use if the user base is JS/TS. | Use if the user base is Go or infra. |
| **CI / GitHub Actions** | `actions/setup-node@v4` + `npm test` | `actions/setup-go@v5` + `go test` |

**Recommendation:** if Neatlogs users are mostly Python + JS, port TypeScript first. If they're Python + Go (infra / K8s), port Go first. The patterns are similar enough that the second port will be 2-3x faster than the first.

---

## 24. What you do NOT need to port

Some things in the Python reference are intentionally Python-specific and don't need a port equivalent:

- **The `dataclass(frozen=True)` decorator.** TS uses `readonly` and `as const`. Go uses value types and explicit constructors. The port is structural, not syntactic.
- **The `argparse` module.** TS uses `parseArgs`, Go uses `flag`. The flag semantics are the same; the syntax differs.
- **The Python `frozenset` for `OTEL_GENAI_LLM_OPERATIONS`.** TS uses `readonly Set<string>` (a regular `Set` is fine — it's only used for membership tests). Go uses `map[string]struct{}`. The semantic ("set of strings") is the same.
- **The Python `defaultdict(list)` for `child_map`.** TS uses `Map<string, Span[]>` with manual `get-or-default`. Go uses `map[string][]string` with manual `get-or-default`. The semantic is the same.
- **The `consult-mmx` review process.** The cross-model review was a Python development-time tool, not a runtime feature. The TS/Go port should have its own review process (a senior TS/Go dev, or a different LLM call).

What you DO need to port exactly:

- The 20 finding codes (§3.1) and their predicates (§4).
- The ordering in §6.1.
- The cycle-detection algorithm in §7.
- The CLI flags and exit codes (§8).
- The `to_dict` "absent when unset" rule (§9).
- The doc-url convention (§11) — in-repo paths only.
- The 6 bugs avoided (§12).

When the port passes all 10 acceptance criteria in §20, ship it.
