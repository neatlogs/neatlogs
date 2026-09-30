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
Documentation fetch failures are retained as evidence gaps. The deterministic
checks never initialize Neatlogs, call a live model provider, export traces, or
query a Neatlogs backend. The scheduled workflow separately calls Gemini for
its advisory assessment when an API key is configured.

## Pull requests

The pull-request workflow is deterministic and does not receive external
service credentials. It builds the SDK wheel, installs it into isolated pip,
uv, and Poetry consumers on Python 3.10 through 3.13, and verifies the public
SDK import surface.

## Scheduled release monitoring

Twice a day, the scheduled workflow:

1. compares the analyzed version lock with PyPI;
2. records dependency, exported API, source-content, adapter-source, and
   official project-documentation evidence;
3. optionally asks Gemini for an advisory impact assessment;
4. creates or updates a review issue and optionally alerts Slack when newer
   releases are detected. It does not create a fix pull request.

The Gemini assessment is unverified advice about possible compatibility risk,
not a confirmed regression. This workflow does not install or smoke test the
newest package versions. It compares PyPI with the tracked analyzed-version
baseline, so the same releases can trigger alerts on later runs until that
baseline is updated. A failed Gemini request leaves the workflow failed while
the evidence artifact and review issue remain available.

Configure these GitHub Actions settings:

- Secret `COMPAT_GEMINI_API_KEY` (optional): a dedicated, quota-limited Gemini
  API key. Without it, deterministic discovery/evidence still runs and the LLM
  step records that it was skipped.
- Variable `COMPAT_GEMINI_MODEL` (optional): model override; defaults to
  `gemini-2.5-flash`.
- Secret `COMPAT_SLACK_WEBHOOK_URL` (optional): a channel-specific Slack
  Incoming Webhook. Without it, Slack notification is skipped.

Organization-level secrets scoped only to the SDK repositories are preferred.
The credentials are used only by the scheduled/default-branch workflow and are
never passed to pull-request jobs. Slack failures are non-blocking; alerts are
sent when PyPI has versions newer than the tracked baseline or when the workflow fails.
