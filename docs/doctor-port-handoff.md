# Trace Doctor — TS & Go Port Handoff

> **Audience:** Engineers porting the `neatlogs-doctor` CLI to TypeScript and Go.
> **Goal:** Reimplement `neatlogs/doctor.py` (1,494 lines, 33 top-level functions) and its full test suite (128 unit tests + 17 integration tests + 3 E2E demo scripts) in your target language, **with no behavior change vs. the Python reference**.
> **Reference implementation:** `neatlogs/doctor.py` in this repo, branch `feat/trace-doctor @ e0b7106`. All `path:line` references below are into that file.
> **Test reference:** `tests/unit/test_doctor.py` (2,577 lines), `tests/integration/test_doctor_e2e.py` (617 lines), `docs/pr20-evidence/doctor_e2e_{demo,dimensions,realsdk}.py`.

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

## 3. Finding codes (the 14 output types)

Every output is a `DoctorFinding` with these required fields:
- `severity: "error" | "warning" | "info"`
- `code: string` (one of the 14 below)
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
The 8 fix_class values map to 4 pipeline stages via `_FIX_CLASS_TO_STAGE`:

| Stage | fix_class |
|---|---|
| `init` | `init_order`, `config`, `pipeline` |
| `instrument` | `instrumentation`, `capture` |
| `span` | `data_integrity`, `attribute` |
| `hierarchy` | `hierarchy` |

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

### 4.16 PR #21 additions — OTel GenAI semconv + token-waste

**`otel-genai-missing` (warning, fix_class=`config`, auto-fixable=False)** — NEW
**Predicate:** An LLM-kind span (neatlogs `kind=llm` OR `gen_ai.operation.name` ∈ {`chat`, `text_completion`, `generate_content`}) lacks `gen_ai.operation.name`. Trace won't be interoperable with OTel GenAI tools (Langfuse, Phoenix, Arize).
**Source:** `_otel_genai_findings()` in `neatlogs/doctor.py`.

**`otel-genai-inconsistent` (info, fix_class=`config`, auto-fixable=False)** — NEW
**Predicate:** A span has BOTH `neatlogs.span.kind=llm` AND `gen_ai.operation.name` but they disagree (e.g., neatlogs says "llm" but OTel says "embeddings" — which is an embedding op, not an LLM op).
**Related codes:** `("otel-genai-inconsistent",)` and `("otel-genai-missing",)` for cross-reference.

**`oversized-prompt` (warning, fix_class=`config`, auto-fixable=False)** — NEW
**Predicate:** A single LLM span's prompt content exceeds 50,000 characters. Counts chars across `gen_ai.input.messages`, `neatlogs.llm.input_messages.*`, `neatlogs.llm.system`, `neatlogs.llm.prompts.*`. Almost certainly a bug (leaked retrieved document, CSV, or log dump).

**`repeated-system-prompt` (info, fix_class=`config`, auto-fixable=False)** — NEW
**Predicate:** The same system prompt content appears in 10+ LLM spans in the same trace. **PII concern — this finding requires `read_prompt_content=True` (off by default).** When off, the check is silently skipped. When on, surfaces a suggestion to enable provider prompt caching.

**`unused-tool-definition` (info, fix_class=`config`, auto-fixable=False)** — NEW
**Predicate:** A tool defined on an LLM span's `gen_ai.tool.definitions` or `neatlogs.llm.tools` is never called in any subsequent span. Reads tool names from OTel `gen_ai.output.messages[*].tool_calls[*].function.name` and neatlogs `neatlogs.llm.tool_calls.*`. No PII concern (tool names only).

**CLI flags added:**
- `--read-prompt-content` (default off): enables the `repeated-system-prompt` check.
- `--emit-fix <finding-code>`: prints a copy-paste-able BEFORE/AFTER snippet for the given code. Does NOT read the log file. Useful for the wizard to show fix snippets without needing a real log.

**The 4 registered fix snippets** (in `_FIX_SNIPPETS` dict in `neatlogs/doctor.py`):
- `init-after-client` — move `neatlogs.init()` to the top of the entry point.
- `missing-span-kind` — add `kind='TOOL'` to `@neatlogs.span` decorator.
- `zero-duration-span` — wrap the body in `try: ... finally: span.end()`.
- `error-status-no-event` — add `span.record_exception(e)` inside the except block.

New codes use the `config` fix_class. These cluster at the **instrument** stage of the pipeline, so a new pipeline-stage-summary would surface them as "instrument-stage" findings.

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
3. **NEW dimensions first** (always run, even on unusual traces):
   - `_init_order_findings`
   - `_attribute_completeness_findings`
   - `_data_integrity_findings`
4. **`rootless-http-only` check** → if true, append finding and **return immediately** (only the new dimensions have already run).
5. **`missing-root-kind` check** → append if no root in ROOT_KINDS.
6. **Hierarchy pathologies** (in this order):
   - `_orphan_parent_findings`
   - `_self_parent_findings`
   - `_duplicate_span_id_findings`
   - `_multiple_roots_findings`
   - `_cycle_findings`
7. **`_agent_without_llm_findings`** (subtree-based).
8. **`_missing_io_findings`**.

**Why this order matters:** the new dimensions must run on EVERY trace shape. Pre-PR-#20, the `rootless-http-only` early-return caused data-integrity and init-after-client to be silently skipped on rootless HTTP traces. This was a real bug (fixed in `9669556`).

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

`neatlogs-doctor [PATH] [--json] [--run-id ID] [--foreign-only]`

| Flag | Effect |
|---|---|
| `PATH` | The span log file. Positional. If omitted, defaults to `$NEATLOGS_LOG_SPANS_FILE` env var. If still unset, the doctor reads from stdin (`-` is the explicit stdin marker). |
| `--json` | Output a JSON report instead of the human-readable report. |
| `--run-id ID` | Only analyze spans whose `session.id` matches. |
| `--foreign-only` | Only return `foreign-instrumentation-detected` findings. |

**Exit code:** 0 if no error-severity findings, 1 otherwise. (Verify with the reference: look at `main()` in `doctor.py`.)

**Default output (human):** A human-readable report. See `format_report()` for the exact format — it includes a header, run/trace counts, and a sorted list of findings with severity icon, code, title, evidence, and suggestion.

**`--json` output:** A single JSON object with `path`, `spans_read`, `trace_count`, `run_count`, `invalid_lines`, `findings`. Each finding is the `to_dict()` form: `{severity, code, title, evidence, suggestion, trace_id?, run_id?, fix_class?, automated_fix_available?, doc_url?, related_codes?}`.

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

---

## 13. Implementation checklist (use this as your todo list)

- [ ] Define the input span-log schema (see §2). Tolerate missing keys with sensible defaults.
- [ ] Implement the status format tolerance (see §2.3).
- [ ] Implement the instrumentation_scope tolerance (string OR dict).
- [ ] Define the `DoctorFinding` / `DoctorReport` types with all 9 fields (severity, code, title, evidence, suggestion, trace_id, run_id, fix_class, automated_fix_available, doc_url, related_codes).
- [ ] Implement `to_dict()` with the "absent when unset" rule (NOT `null` for None fields).
- [ ] Implement run grouping (session.id → trace_id → sentinel).
- [ ] Implement visibility filter (exclude `neatlogs.internal` and `neatlogs.trace.complete`).
- [ ] Implement `_read_spans` (parse JSONL, track invalid lines).
- [ ] Implement the 12 trace-level checks in the exact order in §6.1.
- [ ] Implement the cycle detection algorithm in §7.
- [ ] Implement the missing-io checks (4 sub-types: llm/tool/retriever + the system-only LLM case).
- [ ] Implement the 5 new dimensions (init-order, attribute-completeness, data-integrity, pipeline-stage-summary).
- [ ] Implement the file-level / run-level findings (multi-run-log, run-id-not-found, file-not-found, no-spans, invalid-jsonl, scope-not-preserved).
- [ ] Implement `format_report()` (human-readable) and `--json` (machine).
- [ ] Implement CLI flag parsing (--json, --run-id, --foreign-only, stdin/file).
- [ ] Implement the exit code rule (0 = no error-severity findings, 1 otherwise).
- [ ] Set `doc_url` to in-repo paths (see §11). **Do NOT use `https://docs.neatlogs.com/...` — that domain returns 404.**
- [ ] Set `automated_fix_available=True` for `init-after-client` only. All other findings default to False.
- [ ] Set the `related_codes` filter to use the same `_FIX_CLASS_TO_STAGE` dict as the stage counter (do NOT hardcode init-stage classes).
- [ ] Add tests covering all 14 finding codes (happy + negative per code).
- [ ] Add 17 × 3 = 51 framework coverage tests.
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
| `neatlogs/doctor.py` | 1,494 | The doctor. All 33 top-level functions. |
| `tests/unit/test_doctor.py` | 2,577 | 128 unit tests. |
| `tests/integration/test_doctor_e2e.py` | 617 | 17 integration tests (CLI subprocess runs). |
| `docs/pr20-evidence/doctor_e2e_demo.py` | 21 KB | E2E demo for 5 bugs + 3 enhancements. |
| `docs/pr20-evidence/doctor_e2e_dimensions.py` | 11 KB | E2E demo for 4 new dimensions. |
| `docs/pr20-evidence/doctor_e2e_realsdk.py` | 3 KB | Real SDK roundtrip demo. |
| `skills/neatlogs/references/troubleshooting.md` | — | The in-repo doc that `doc_url` fields point at. |
| `neatlogs/init.py` | 759 | The `neatlogs.init()` function. The `init-after-client` suggestion refers to its `shutdown()` companion. |
| `neatlogs/core/span_processor.py` | — | Where the SDK writes the span log files the doctor reads. Reference for the exact JSON shape. |

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

1. **All 14 finding codes** (the 10 pre-existing + 4 new dimensions) fire correctly across the existing test suite.
2. **All 5 bugs** listed in §12 are avoided (the port implements the fixes from day one).
3. **All 51 framework coverage tests** (17 × 3) pass.
4. **Performance:** 100K spans in <1s on comparable hardware.
5. **CLI:** `--json`, `--run-id`, `--foreign-only`, file/stdin, exit code 0/1.
6. **Backward compat:** the JSON report for an old log file (no `instrumentation_scope`, no new fields on findings) is the same shape as the Python reference's output for the same input.
7. **Doc URLs:** point to in-repo paths, never to `docs.neatlogs.com`.
8. **Suggestion text:** the `init-after-client` suggestion does not mention `force_reload=True`.

When you can pass all 8 criteria, the port is feature-complete and you can ship.
