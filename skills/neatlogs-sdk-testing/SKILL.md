---
name: neatlogs-sdk-testing
description: Plan and execute contract-driven testing of the published and local Neatlogs Python SDK without modifying SDK source while tests are running.
---

# Neatlogs Python SDK testing

Use this skill for SDK validation, release qualification, lifecycle characterization, provider integration testing, or published-package developer-experience checks.

## Safety and scope

- Start from a clean worktree and report its branch and commit.
- Do not edit SDK source, tests, lockfiles, or documentation during an execution pass. Put generated consumers and artifacts in a temporary directory.
- Never push, publish, or open a pull request unless separately authorized.
- Load credentials only from an ignored local file or process environment. Never print credentials or pass them in command arguments.
- Use the exact requested provider, model, endpoint, package version, Python version, and package manager. Do not silently fall back.
- Write the telemetry contract before making a provider call. Do not weaken it after failure.
- Stop live execution when credentials, runtime, dependency, endpoint, or model diagnostics fail. Report configuration needs without requesting secrets in chat.

## Required PASS contract

A case passes only when all applicable gates pass:

1. The application returns the intended value or intended error.
2. Local normalized telemetry matches exact span names, kinds, counts, fields, status, and parent relationships.
3. `flush()` and `shutdown()` complete with the expected result.
4. The exact captured trace ID is retrieved from the configured backend.
5. Persisted spans match the local contract and all belong to that trace.

Do not treat an HTTP success, provider response, generated trace ID, or dashboard link alone as PASS. Show a dashboard URL only after exact-ID persistence succeeds, and describe dashboard inspection as manual.

## Test modes

### Clean consumer

- Create an empty temporary project.
- Install the published wheel/sdist using each supported Python 3.10–3.13 environment.
- Test an unconstrained install first. Record resolver or native-build failures before applying a temporary compatibility constraint.
- Copy the smallest README example without importing repository source.
- Test pip and uv consumer flows; add Poetry only when release scope requires it.

### Local source

- Use an isolated virtual environment.
- Install the current branch as a package instead of injecting repository paths.
- Run unit and integration suites separately so optional-dependency skips are visible.

### Live E2E

- Use bounded deterministic prompts and tool inputs.
- Prefer fixed markers such as `NEATLOGS_OK`, low temperature, and small token limits.
- Record package/provider versions, trace ID, persistence result, span count, and elapsed time.

## Initialization and lifecycle matrix

Run each global-state case in a fresh subprocess:

- Default and explicit initialization.
- Missing, empty, whitespace, and invalid ingestion credentials.
- Environment-only and explicit endpoint configuration.
- Initialization twice and concurrent initialization.
- Reinitialization after shutdown.
- Flush before initialization, twice, and after shutdown.
- Shutdown before initialization and twice.
- Span creation during and after shutdown.
- Shutdown with active children.
- Cached provider wrapper after shutdown.
- Normal exit without explicit flush.
- SIGINT and SIGTERM subprocess behavior.
- Export disabled.
- Sampling at `0`, `1`, values just inside the range, and invalid values below/above the range.

For every case, distinguish API return values from actual exporter and persistence outcomes.

## Python-specific coverage

- Sync, async, generator, and async-generator decorators.
- Threads, task propagation, cancellation characterization, and secondary-client isolation.
- Inputs containing nulls, empty values, Unicode, bytes, recursive objects, Pydantic models, dataclasses, and unserializable values.
- Global and per-span masking, PII modes, structured logs, and exception redaction.
- Import before/after `init()`, absent optional dependencies, incompatible versions, and double wrapping versus auto-instrumentation.
- Provider non-streaming, full streaming, early termination, tools, errors, retries, structured output, embeddings, and retrieval where supported.

## Required report

For each case report: package/runtime versions, setup, expected contract, application outcome, lifecycle return values, exporter evidence, persistence status, trace ID, dashboard URL when valid, result (`PASS`, `FAIL`, `CHARACTERIZATION`, or `NOT APPLICABLE`), and defect notes. Include skipped integrations and the precise reason.

