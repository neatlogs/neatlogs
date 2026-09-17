# Compatibility automation

This directory defines the integrations, package-manager matrix, supported
versions, and cross-integration contracts exercised by the compatibility
workflows.

## Scope source of truth

The inventory is limited to integrations documented for the Python SDK in
`neatlogs-docs`, including its Python supported-libraries table. Coding agents
owned by separate repositories are intentionally excluded.

These workflows analyze real published package contents, APIs, dependency
graphs, and the relevant adapter source. They never initialize Neatlogs, call a
live model provider, export traces, or query a Neatlogs backend.

## Pull requests

The pull-request workflow is deterministic and does not receive external
service credentials. It builds the SDK wheel, installs it into isolated pip,
uv, and Poetry consumers on Python 3.10 through 3.13, and verifies the public
SDK import surface.

## Scheduled release monitoring

Twice a day, the scheduled workflow:

1. compares the analyzed version lock with PyPI;
2. records package metadata and wheel/sdist manifest changes as deterministic
   evidence;
3. optionally asks Gemini for an advisory impact assessment;
4. updates a GitHub issue and optionally alerts Slack when review is needed.

The Gemini assessment is advisory only. It cannot change a compatibility
verdict or make a workflow pass.

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
sent only for newly discovered releases or workflow failures.
