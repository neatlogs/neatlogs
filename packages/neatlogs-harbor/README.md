# neatlogs-harbor

Send [Harbor](https://github.com/harbor-framework/harbor) agent-eval jobs to Neatlogs: each task becomes a dataset case, each (agent, model, dataset) becomes a run, and every trial arrives with its reward, tokens, cost, timings, error class and a full trace.

```bash
uv tool install harbor --with neatlogs-harbor
harbor run -d terminal-bench@2.0 -a codex -m openai/gpt-5.6-sol --plugin neatlogs
```

When it runs:

- **Job start:** the job's tasks are published as a dataset version.
- **Each trial end:** the trial's trace is sent.
- **Job end:** the runs are registered, every trial's output is uploaded, the runs are completed and compared with the previous run, and a summary is printed.

Runs are registered at job end because a run accepts outputs for one hour after registration; registering late means the window covers the upload, not the whole job. The job end runs the same code as `neatlogs harbor import`.

The plugin never fails a Harbor run. Anything it cannot send waits in `jobs/<job>/neatlogs-sync.json`, and `neatlogs harbor sync jobs/<job>` sends it later.

## Setup

The plugin reads these from the Harbor process environment. Harbor does not pass them into task sandboxes.

| Variable | What it is | Used for |
|---|---|---|
| `NEATLOGS_API_KEY` | The project's ingest key, the same one the Neatlogs SDK uses | Traces (OTLP) |
| `NEATLOGS_TOKEN` | A service-account token with `evaluation:write` | Datasets and runs |
| `NEATLOGS_PROJECT_ID` | The project. Optional when the token can see one project | Datasets and runs |
| `NEATLOGS_HARBOR_FORM_VERSION_ID`, `NEATLOGS_HARBOR_EVALUATOR_IDS` | The evaluation form version and evaluators that grade the runs | Runs |
| `NEATLOGS_HOST` | App origin. Default `https://app.neatlogs.com` | Datasets and runs |
| `NEATLOGS_ENDPOINT` | Ingest origin. Default `https://ingest.neatlogs.com` | Traces |

Two credentials and a grading form are needed today because the public runs API only accepts service-account tokens and requires a grading form for every run. With only `NEATLOGS_API_KEY`, traces flow and the outputs wait in the ledger until the rest is set.

## What it sends

| Harbor | Neatlogs |
|---|---|
| Dataset (`terminal-bench@2.0`, a local tasks folder, or the task name for `-t`) | A dataset. Each distinct set of task digests (`lock.json` → `task.digest`) gets its own, so an edited task never mixes with old results |
| Task | A case. Input is `instruction.md` from the task folder; metadata holds the digest. No expected output: the verifier decides |
| Agent × model × dataset in a job | A run named `agent · model`, for example `codex · gpt-5.6-sol` |
| Trial (task × attempt) | A run output for that case and attempt. Attempts are numbered by start time, since Harbor records no attempt number |
| `result.json` rewards, tokens, cost, phase timings | Output `metrics`: `reward`, `reward.<key>`, `solved`, `rewardValid`, `inputTokens`, `outputTokens`, `totalTokens`, `cacheTokens`, `costUsd`, `durationMs`, `latencyMs` and per-phase milliseconds |
| `exception_info` + Harbor's error policy | `errorClass` in the output: `agent` (timeouts, refusals: graded and counted), `grader` (verifier failed) or `infrastructure` (sandbox, provider, Harbor). Rewards are only reported for trials with a valid reward, so infrastructure failures stay out of averages as they do in Harbor |
| `agent/trajectory.json` (ATIF) | One trace per trial: a `harbor.trial` root, a span per phase, and the agent's model and tool calls under the agent phase, converted by Harbor's own `harbor-atif2otel`. Agents without a trajectory still get the phase trace |
| `verifier/reward-details.json`, `ctrf.json`, test output tail | Attached to the output under `verifier` (turn off with `--pk attachments=none`) |

Traces carry `neatlogs.origin=simulation` and `deployment.environment=test`.

At job end the plugin compares each run with the previous Harbor run on the same dataset version (the one tagged `harbor-latest`), moves the tag forward, and prints a summary:

```
Neatlogs summary
  codex · gpt-5.6-sol  ·  terminal-bench@2.0
    solved        37 / 89 trials  41.6%
    pass@3        52.8%  ·  all 3 attempts solved 31.5%
    cost          $41.20  ·  $1.11 per solved trial
    no reward     2 trial(s), left out as Harbor does
    vs previous   grading in progress; the comparison fills in on the dataset page
    open          https://app.neatlogs.com/experiments/datasets/…
```

## Jobs that already ran

```bash
neatlogs harbor import jobs/          # every job in the folder, oldest first
neatlogs harbor sync jobs/<job>       # what one job still has pending
neatlogs harbor import jobs/ --dry-run
```

`neatlogs harbor …` works when the Neatlogs SDK is installed; `neatlogs-harbor …` is the same command without it.

Every record uses ids derived from Harbor's job id, task name and trial name (uuid5) and a matching `Idempotency-Key`, so the plugin, `import` and `sync` reconcile instead of duplicating. Traces use a trace id derived from the trial id; the importer checks whether a trace already exists before sending it.

## Redaction

Before anything is uploaded the plugin removes token-shaped strings (provider API keys, GitHub, AWS, Google, Slack, Stripe and Hugging Face tokens, JWTs, bearer tokens, private keys, `password=…` assignments) and the exact values of secret-looking environment variables in the Harbor process (`*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_KEY`). Add patterns with `--pk redact='<regex>;;<regex>'`.

## Options

Plugin kwargs (`--pk key=value`) win over environment variables.

| Kwarg | Env | Default |
|---|---|---|
| `name` | `NEATLOGS_HARBOR_RUN_NAME` | `agent · model` |
| `baseline` | `NEATLOGS_HARBOR_BASELINE` | `tag:harbor-latest` (previous Harbor run on the dataset version) |
| `redact` | `NEATLOGS_HARBOR_REDACT` | none |
| `attachments` | `NEATLOGS_HARBOR_ATTACHMENTS` | `default`; `none` sends no verifier files |
| `traces` | `NEATLOGS_HARBOR_TRACES` | `true` |
| `api_key`, `token`, `project_id`, `host`, `endpoint`, `form_version_id`, `evaluator_ids` | as in Setup | |

## Current limits (Phase 1)

These come from the public runs API as it is today:

- A run accepts outputs for one hour after it is registered, which is why runs are registered at job end. A run with an output that failed to upload is left open, so `neatlogs harbor sync` can still fill it within that hour.
- A run holds at most 100 tasks and 10 attempts per task, which fits benchmarks the size of Terminal-Bench. Larger jobs still send traces and the dataset; extra attempts are recorded as skipped.
- Requests are capped at 64 KiB, so long task instructions are shortened in the dataset case. The full text is in each trial's trace.
- Comparisons are computed from the grading form's evaluator grades. Rewards are stored as metrics but not yet compared; verifier scores as first-class scores are Phase 2.
- The public API cannot add cases to an existing dataset, so a changed task set publishes a new dataset under the same name.

## Development

```bash
cd packages/neatlogs-harbor
python -m pip install -e ".[test]"
python -m pytest -q
```

The tests run against an in-memory Neatlogs (`tests/fake_neatlogs.py`) that checks every request and response against `tests/fixtures/contracts/public-api.json`. That file is the part of the Neatlogs public OpenAPI document this package uses. Refresh it with `python tests/fixtures/contracts/extract.py <public-openapi.json>`.
