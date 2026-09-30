# Compatibility automation

This directory defines the integrations, package-manager matrix, supported
versions, and cross-integration contracts exercised by the compatibility
workflows.

## Scope source of truth

The inventory is limited to integrations documented for the Python SDK in
`neatlogs-docs`, including its Python supported-libraries table. Coding agents
owned by separate repositories are intentionally excluded.

These workflows analyze real published package contents, exported APIs,
dependency graphs, changed source excerpts, the relevant adapter source, and
the official project documentation URLs declared for every integration.
Documentation fetch failures are retained as evidence gaps. The pull-request
checks do not initialize Neatlogs; the scheduled activation checks do so with
export disabled. Neither check calls a live model provider or a Neatlogs
backend. The scheduled workflow separately calls Gemini when an API key is
configured.

## Pull requests

The pull-request workflow is deterministic and does not receive external
service credentials. It builds the SDK wheel, installs it into isolated pip,
uv, and Poetry consumers on Python 3.10 through 3.13, and verifies the public
SDK import surface.

## Scheduled release monitoring

Twice a day, the scheduled workflow:

1. compares the analyzed version lock with PyPI;
2. builds the SDK wheel once and, for each changed package/integration pair,
   compares the tracked and latest published versions in separate environments;
3. checks exact-version installation, dependency consistency, and Neatlogs
   instrumentation activation without calling a provider;
4. independently records exported API, source, adapter, and documentation
   evidence and optionally asks Gemini for an advisory assessment;
5. creates or updates a review issue, then asks Gemini for a concrete fix
   decision on one package/integration candidate per run;
6. validates a bounded adapter-source patch in a separate job without model
   or write credentials, and opens or reuses a draft PR only if validation
   passes. It never merges the PR. Slack reports the result.

The Gemini assessment is unverified advice about possible compatibility risk,
not a test result. A latest-version pass covers only this activation smoke
check, not the full SDK behavior. A baseline-passing/latest-failing check is
reported as an activation smoke regression; other failures need triage.
Blocked and missing checks are explicitly reported. The same releases can
trigger later alerts until the analyzed-version baseline is updated. A failed
Gemini request is reported as an unavailable advisory while the deterministic
results, evidence, and review issue remain available.

A high advisory score alone cannot create a PR. The separate fix proposal must
name the changed package and integration, cite evidence, and replace an exact,
unique excerpt in a current SDK adapter. Workflow, configuration, and arbitrary
file edits are rejected. Validation runs compatibility automation tests,
focused SDK tests, and an exact latest-version activation check. When Gemini
supplies a regression test, it must fail on the original SDK and pass after
the patch. A draft without that focused red/green proof is explicitly marked
unverified for human review. One candidate is attempted per run, prioritizing
baseline-passing activation failures. Other candidates are listed as deferred.
An existing PR for the
same package/version/integration is reused rather than overwritten.

Configure these GitHub Actions settings:

- Secret `COMPAT_GEMINI_API_KEY` (optional): a dedicated, quota-limited Gemini
  API key. Without it, deterministic discovery/evidence still runs and the LLM
  step records that it was skipped.
- Variable `COMPAT_GEMINI_MODEL` (optional): model override; defaults to
  `gemini-2.5-flash`.
- Secret `COMPAT_SLACK_WEBHOOK_URL` (optional): a channel-specific Slack
  Incoming Webhook. Without it, Slack notification is skipped.
- Secret `COMPAT_PR_TOKEN` (recommended for draft PR creation): a narrowly
  scoped GitHub App token or PAT with repository contents and pull-request write
  access. The workflow falls back to `GITHUB_TOKEN`, but PRs created with that
  token may not trigger normal pull-request CI. Repository Actions settings must
  allow workflows to create pull requests.

Organization-level secrets scoped only to the SDK repositories are preferred.
Model and write credentials are confined to separate scheduled jobs and are
never passed to generated-code validation or pull-request CI jobs. Slack
failures are non-blocking; alerts are sent when PyPI has versions newer than
the tracked baseline or when the workflow fails.
