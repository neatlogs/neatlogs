# Copy-to-IDE Prompt Structure

> How Neatlogs structures suggestions for AI coding agents when users click "Copy to IDE".

## Overview

When a user opens a suggestion in Neatlogs and clicks **Copy to IDE**, the system serializes the suggestion into a structured prompt optimized for AI coding agents (Cursor, Claude Code, GitHub Copilot, Windsurf, Cline, etc.). The goal is to give the agent everything it needs to understand and fix the issue in a single paste — no back-and-forth required.

This document explains the structure, the reasoning behind each section, and the principles that guide the format.

---

## Prompt Structure

The generated prompt follows a four-section architecture:

```
┌─────────────────────────────────────┐
│  1. TASK        (what to do)        │
├─────────────────────────────────────┤
│  2. CONTEXT     (what went wrong)   │
├─────────────────────────────────────┤
│  3. IMPLEMENTATION (how to fix it)  │
├─────────────────────────────────────┤
│  4. REFERENCE   (supporting data)   │
└─────────────────────────────────────┘
```

### Section 1: Task

**Purpose:** Frame the prompt as an actionable instruction, not passive documentation.

Contains:
- Urgency directive based on severity (critical/high/medium)
- Category hint explaining what part of the system is affected (prompt, tool, model, config, cost)
- Pattern description explaining the detected failure mode
- Explicit instruction to read context, implement the fix, and verify

**Why this matters:** Research from Anthropic's context engineering and Cursor's agent best practices shows that AI agents perform significantly better when given an explicit task framing upfront. Without it, agents may summarize the content rather than act on it. The task section converts a "here's information" document into a "do this thing" instruction.

### Section 2: Context

**Purpose:** Provide the diagnostic information the agent needs to understand the problem.

Contains:
- Issue title and classification (severity, category, pattern, workflow)
- Root cause analysis
- Evidence and reasoning

**Why this matters:** Following Sentry's autofix approach and CodeRabbit's agent handoff format, context should be presented after the task instruction but before implementation steps. This ordering (task → context → steps) matches how humans naturally process instructions and yields up to 30% better results in long-context scenarios (per Anthropic's research).

### Section 3: Implementation

**Purpose:** Give the agent concrete steps to follow and criteria to verify against.

Contains:
- Ordered implementation steps
- Constraints and risk assessment
- Verification checklist (checkbox format)

**Why this matters:** Coding agents work best with decomposed, sequential instructions. The verification checklist serves as a self-check mechanism — agents like Claude Code and Cursor will actually iterate on their solution until all checkboxes can be satisfied. This pattern is inspired by CodeRabbit's "phases + tasks" structure and the CIF (Context-Intent-Format) framework.

### Section 4: Reference

**Purpose:** Provide supporting data that may be needed during implementation but isn't part of the core instruction flow.

Contains:
- Linked trace IDs (for inspecting actual failures)
- Confidence breakdown (helps agent prioritize which aspects to focus on)
- Trigger metadata (signal type, source)
- Additional metadata (arbitrary JSONB from the detection pipeline)

**Why this matters:** Reference data is placed last because it's supplementary. Anthropic's research on long-context ordering shows that placing the query/task at the beginning and reference data at the end produces better results than interleaving them. Agents can look back at this section when needed without it cluttering the primary instruction flow.

---

## Design Principles

### 1. Task-First Framing

Every prompt starts with an explicit instruction. This is the single most impactful change from a "data dump" format to an "agent-ready" format.

**Inspiration:** Anthropic's context engineering guide emphasizes that system prompts should use "simple, direct language" with explicit task definitions. CodeRabbit's agent handoff format always includes a machine-readable "Agent Prompt" section distinct from the human-readable review.

### 2. Structured Sections with Clear Boundaries

Each section has a clear H1 heading (`# Task`, `# Context`, `# Implementation`, `# Reference`). This creates unambiguous boundaries that prevent the agent from confusing instructions with context.

**Inspiration:** Anthropic recommends XML tags or markdown headers as "first-class best practice" for organizing context. When instructions and data blur together, models have to guess where boundaries are — that's where errors occur.

### 3. Progressive Detail (Inverted Pyramid)

The most actionable information comes first (what to do), followed by understanding (why), then steps (how), then reference (supporting data). An agent that stops reading early still has enough to attempt the fix.

**Inspiration:** Sentry's autofix agent architecture processes issues in stages: problem discovery → planning → execution → review. Our prompt mirrors this flow so the agent can process it sequentially.

### 4. Explicit Verification Criteria

Every prompt includes a verification checklist when available. This gives the agent a concrete definition of "done" and enables self-checking behavior.

**Inspiration:** Claude Code's best practices emphasize that prompts should include "explicit verification loops." Cursor's agent documentation recommends ending with cleanup/verification requests. The checkbox format (`- [ ]`) is universally understood by coding agents.

### 5. Contextual Category and Pattern Hints

Rather than just labeling an issue as "prompt" or "tool_contract_misuse", the prompt includes a natural-language explanation of what that means. This helps agents that may not be familiar with Neatlogs-specific terminology.

**Inspiration:** Anthropic's tool engineering guide recommends "contextual relevance over flexibility" — prefer human-readable descriptions over technical identifiers.

### 6. Minimal Noise

The prompt excludes:
- Internal IDs that don't help the agent (org_id, project_id)
- Status/workflow metadata (draft, open, in_progress) — irrelevant to fixing
- Timestamps (created_at, updated_at) — not actionable
- Sidebar comments — may contain off-topic discussion

**Inspiration:** The "minimal but sufficient" principle from Anthropic's prompt engineering: start with what's needed, add only what improves outcomes. Every token in the prompt should earn its place.

---

## Failure Pattern Descriptions

Each triage label maps to a specific natural-language description in the prompt:

| Triage Label | Description in Prompt |
| --- | --- |
| `prompt_regression` | A previously working prompt is now producing worse results |
| `hallucination` | The model is generating factually incorrect or fabricated content |
| `tool_contract_misuse` | Tool calls are being made with incorrect arguments or in wrong contexts |
| `retrieval_miss` | Relevant context is not being fetched or is being ignored |
| `latency_anomaly` | Response times have degraded significantly |
| `cost_anomaly` | Token consumption or API costs have spiked unexpectedly |
| `orchestration_inefficiency` | The agent workflow has redundant or unnecessary steps |
| `guardrail_bypass` | Safety or validation checks are being circumvented |
| `user_frustration_signal` | End users are showing signs of dissatisfaction with responses |
| `silent_degradation` | Quality is declining without obvious errors |

---

## Category Descriptions

| Category | Description in Prompt |
| --- | --- |
| `prompt` | The LLM instructions need adjustment |
| `tool` | A tool call or its schema needs fixing |
| `model` | The model selection or parameters need tuning |
| `config` | System settings or environment config needs updating |
| `cost` | Reduce token usage or unnecessary API calls |

---

## Example Output

For a critical prompt regression issue, the generated prompt looks like:

```markdown
# Task

Fix the following critical issue immediately. This is a prompt engineering issue — the LLM instructions need adjustment. The detected pattern is a prompt regression — a previously working prompt is now producing worse results. Read the full context below, then implement the fix following the steps in the Implementation section. Verify your changes against the checklist before committing.

# Context

## System prompt causes JSON parsing failures after v2.3 update

**Severity:** Critical | **Category:** Prompt | **Pattern:** Prompt regression
**Workflow / Agent:** order-processing-agent
**Confidence:** 87% | **Source:** Auto-triggered detection

## Root Cause

The system prompt was updated in v2.3 to include few-shot examples, but the examples contain unescaped curly braces that conflict with the JSON output format instruction. The model now intermittently produces malformed JSON because it treats the example braces as part of the output template.

## Evidence

- 23 out of 50 traces in the last 4 hours show JSON parse errors
- All failures occur in the `extract_order_details` tool call
- The error pattern started exactly when commit abc123 was deployed
- Previous version (v2.2) had 0% failure rate on the same inputs

# Implementation

## Steps

1. Escape all curly braces in the few-shot examples using double-brace syntax `{{` and `}}`
2. Add an explicit output format reminder after the examples section
3. Add a JSON validation step before returning the tool call result

## Constraints

- Risk: Low — only affects prompt formatting, no logic changes
- Rollback: Revert to v2.2 system prompt if fix doesn't resolve within 1 hour

## Verification

After implementing the fix, confirm each of the following:

- [ ] All few-shot examples have properly escaped braces
- [ ] The output format instruction appears after the examples
- [ ] Run 10 test inputs through the agent and confirm valid JSON output
- [ ] No regression in response quality for non-JSON outputs

# Reference

## Linked Traces

These trace IDs contain the observed failures. Use them to inspect the actual execution in Neatlogs:

- `trace-abc-123`
- `trace-def-456`
- `trace-ghi-789`

## Confidence Breakdown

| Factor | Score |
| --- | --- |
| Evidence strength | 92% |
| Pattern consistency | 85% |
| Detection alignment | 88% |
| Historical match | 79% |
| User signal weight | 90% |
| Expected impact | 87% |

---
*Generated by [Neatlogs](https://neatlogs.com) · Suggestion 550e8400-e29b-41d4-a716-446655440000 · 2025-01-15T10:30:00.000Z*
```

---

## References & Inspirations

| Source | Key Takeaway | Applied As |
| --- | --- | --- |
| [Anthropic — Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Use structured sections with clear boundaries; data first, query at end; explicit task definition | Section architecture, H1 boundaries, task-first framing |
| [Anthropic — Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | Prefer human-readable descriptions over technical identifiers | Category/pattern natural-language hints |
| [Cursor — Agent Best Practices](https://cursor.com/blog/agent-best-practices) | Include file references, paste error logs, reference existing code | Evidence section, linked traces |
| [CodeRabbit — Agent Handoff](https://docs.coderabbit.ai/plan/agent-handoff) | Summary → Research → Design → Phases → Tasks → Agent Prompt | Four-section progressive structure |
| [CodeRabbit — Plan Refinement](https://docs.coderabbit.ai/plan/plan-refinement) | Break work into phases; hand off one phase at a time | Steps as ordered list |
| [Sentry — AI Autofix Architecture](https://blog.sentry.io/how-sentrys-ai-autofix-changed-my-mind-about-ai-agents/) | Issue summary + error context + code context + task instructions | Context section structure |
| [Datadog — Bits AI Dev Agent](https://www.datadoghq.com/blog/bits-ai-dev-agent/) | Ground fixes in production observability data | Linked traces, confidence breakdown |
| [Claude Code — CLAUDE.md Best Practices](https://arize.com/blog/claude-md-best-practices-learned-from-optimizing-claude-code-with-prompt-learning/) | Under 60 lines for rules; explicit verification loops | Verification checklist, concise task section |
| CIF Framework (Context-Intent-Format) | Specify context, state intent explicitly, define output format | Three-part structure within each section |

---

## Implementation Location

The serialization logic lives in the frontend:

```
neatlogs-app/src/app/(app)/suggestions/_libs/serialize-suggestion-for-ide.ts
```

This is a pure function that takes a `SuggestionDetail` object and returns a markdown string. It has no side effects and no API calls — all data is already available from the suggestion detail query.

---

<!-- ──────────────────────────────────────────────────────────────────────────
## MCP Integration (Coming Soon)

> **Status:** Not yet exposed publicly. This section is reserved for when the
> Neatlogs MCP server is fully available.

When the Neatlogs MCP (Model Context Protocol) server is available, coding agents
can fetch live observability data directly instead of relying solely on the
static prompt content. This enables richer, real-time context during fix
implementation.

### How It Works

1. **Authentication:** The user connects their coding agent to Neatlogs MCP
   using their project API key:
   ```json
   {
     "mcpServers": {
       "neatlogs": {
         "url": "https://mcp.neatlogs.com",
         "headers": {
           "Authorization": "Bearer <YOUR_NEATLOGS_API_KEY>"
         }
       }
     }
   }
   ```

2. **Available Tools:** Once connected, the agent can call:
   - `neatlogs_get_trace` — Fetch full trace details (spans, events, attributes)
   - `neatlogs_search_traces` — Search traces by workflow, time range, or error pattern
   - `neatlogs_get_suggestion` — Fetch the full suggestion with all metadata
   - `neatlogs_get_detections` — List active detections for the project
   - `neatlogs_get_logs` — Fetch logs correlated with a trace or time window

3. **Enhanced Workflow:** With MCP connected, the Copy-to-IDE prompt can include
   a hint like:
   ```
   Note: This project has Neatlogs MCP connected. You can fetch live trace data
   using the `neatlogs_get_trace` tool with the trace IDs listed in the Reference
   section for deeper inspection.
   ```

4. **Benefits:**
   - Agent can inspect actual span-level data from linked traces
   - Agent can search for similar failures across the project
   - Agent can verify its fix by checking if new traces pass detection rules
   - No need to manually copy-paste trace data into the prompt

### Configuration for Popular Agents

**Cursor / Windsurf (MCP config in `.cursor/mcp.json`):**
```json
{
  "mcpServers": {
    "neatlogs": {
      "url": "https://mcp.neatlogs.com",
      "headers": {
        "Authorization": "Bearer <YOUR_NEATLOGS_API_KEY>"
      }
    }
  }
}
```

**Claude Code (MCP config in `.claude/mcp.json`):**
```json
{
  "mcpServers": {
    "neatlogs": {
      "command": "npx",
      "args": ["@neatlogs/mcp-server"],
      "env": {
        "NEATLOGS_API_KEY": "<YOUR_NEATLOGS_API_KEY>"
      }
    }
  }
}
```

**VS Code + Copilot (settings.json):**
```json
{
  "mcp.servers": {
    "neatlogs": {
      "url": "https://mcp.neatlogs.com",
      "headers": {
        "Authorization": "Bearer <YOUR_NEATLOGS_API_KEY>"
      }
    }
  }
}
```

### Prompt Modification When MCP Is Detected

When the system detects that a user's project has MCP configured (future feature),
the Task section of the prompt will append:

```
You have access to Neatlogs MCP tools. Use `neatlogs_get_trace` to inspect the
linked trace IDs for detailed span-level data before implementing your fix.
After implementing, use `neatlogs_search_traces` to verify no similar failures
exist in recent traces.
```

This keeps the prompt self-contained for users without MCP while giving enhanced
instructions to those who have it configured.

────────────────────────────────────────────────────────────────────────────── -->

---

## Changelog

| Date | Change |
| --- | --- |
| 2025-07-15 | Initial version — four-section architecture (Task, Context, Implementation, Reference) |
