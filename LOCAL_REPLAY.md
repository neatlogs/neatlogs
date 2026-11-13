# Run your agent with saved results

Local replay runs your own Python code against a saved trace or pinned dataset. You can reuse saved tool results while trying a different model, prompt or agent change.

The Python SDK executes this workflow itself. Node and the TypeScript CLI are not required.

## Mark JSON tools

```python
import neatlogs

neatlogs.init(instrumentations=["openai", "google_genai"])

@neatlogs.span(kind="TOOL", tool_name="lookup_order", replay=True)
def lookup_order(order_id: str):
    return read_order(order_id)
```

Tool matching uses the tool name and exact JSON arguments, including Python defaults. Repeated identical calls consume their saved results in order. Async functions work with `replay=True`. A regular function that returns an awaitable uses `replay="awaitable"`.

Tools must return JSON values. Generator results and arbitrary Python classes cannot be restored as JSON tools. Legacy MCP string results whose original return shape is ambiguous are skipped.

## Choose what to reuse

- `tools_only`: tools reuse matching saved results; models execute normally. This is the default.
- `until_span`: complete saved model and tool calls can be reused before the selected checkpoint. The checkpoint itself executes normally, followed by the rest of the agent.

The checkpoint must exist in the verified recording. A missing or unsupported match switches the remaining execution to live calls, including tools. It never silently substitutes an approximate result.

Every replay span has origin `simulation`, its source trace, mode and result status. Uploaded attempts also carry their run, case and trial. Reports show `HIT`, the first `MISS`, live-call counts and generated trace IDs. A cached model response retains its original SDK data for application code, while exported billed token and cost metrics are zero.

## Run one saved file

Create `.neatlogs/agent.yaml`:

```yaml
command: ["python", "agent.py"]
timeout_seconds: 300
```

Then run:

```sh
python -m neatlogs simulate local --recording saved-case.json --json
```

The file can be a frozen case with saved spans or a version-one replay recording. The agent reads its input as one JSON value from stdin. Frozen input may itself be a JSON string; the CLI preserves that value exactly.

The child agent must call `neatlogs.init()` so it can load the explicit replay environment, then call `neatlogs.shutdown()` after its root workflow ends. Local fixtures can use `init(disable_export=True, instrumentations=[])` without ingestion or provider credentials.

Optional input mapping uses saved case fields without parsing or guessing their meaning:

```yaml
command: ["python", "agent.py"]
input_from_trace:
  question: root.input.question
```

This mapping requires `root.input` to be an object containing `question`.

## Run pinned dataset examples

Set `NEATLOGS_HOST`, `NEATLOGS_PROJECT_ID` and `NEATLOGS_TOKEN` for dataset access. The public API token is separate from the agent's ingestion API key and provider credentials.

```sh
python -m neatlogs simulate local \
  --dataset DATASET_ID --version VERSION_ID \
  --case CASE_ID --agent .neatlogs/agent.yaml --json
```

Repeat `--case` for up to 100 different examples. Use `--trials 1..10`, `--label`, or `--param name=value` as needed. Parameters are available to your agent through `NEATLOGS_REPLAY_PARAMS`; your code decides how to apply them.

To save and grade the new traces, add:

```sh
--upload --form FORM_VERSION_ID --evaluator EVALUATOR_ID
```

Repeat `--evaluator` for more graders. Upload requires exactly one exported workflow trace per attempt. It waits for trace delivery before accepting the output. A failed child command is never uploaded.

## Run the whole dataset and resume

```sh
python -m neatlogs simulate local \
  --dataset DATASET_ID --version VERSION_ID \
  --all --state .neatlogs/run-state.json --batch-size 25 \
  --agent .neatlogs/agent.yaml --json
```

The CLI pins every case and revision before execution. It supports up to 2,000 examples, in batches of at most 100. Add the upload and grading flags above to create saved batch runs.

Continue with the same project, API origin, working directory and agent settings:

```sh
python -m neatlogs simulate local --resume .neatlogs/run-state.json --json
```

Accepted attempts never execute again. An interrupted upload retries with its saved idempotency key. A command whose outcome is unknown stops resume before another agent command runs. Inspect that attempt before starting another run.

The state file uses the same version-one protocol as the TypeScript SDK. It is private, bounded and locked during use. It stores settings hashes, pinned IDs, progress and activity reports, without recording the API token, command text, raw case contents or child output.

## Provider support

Automatic model replay and raw capture support these wrapped, non-streaming operations with plain JSON request arguments:

| Provider | Python operation |
| --- | --- |
| OpenAI | `chat.completions.create` |
| OpenAI | `responses.create` |
| Google | `models.generate_content` |

Both sync and async wrappers return actual installed SDK response classes. The offline capability tests use OpenAI 2.15.0 and google-genai 1.57.0, including text getters, function calls and signed Google parts.

A model recording requires the complete, hash-verified raw capture saved by a supported wrapper. Normalized text or an old trace without that capture is insufficient. Streaming, parse helpers, raw-response helpers, transport options, typed request/config objects, media, masked captures and unsupported SDK shapes are not restored as model cache hits. Unsupported calls encountered through the listed hooks execute normally and make checkpoint replay switch live. Other operations need an explicit `replay_model()` adapter to participate in replay reports.

Custom masking removes the duplicate raw model capture. Incomplete, redacted, truncated or failed saved executions cannot populate the replay cache.

## Programmatic use

```python
from neatlogs import ReplaySession, get_replay_status

with ReplaySession(recording, mode="tools_only") as replay:
    answer = run_agent(question)

print(replay.get_status())
```

Use `async with` or `session.arun()` for async entry points and awaitable factories. Sessions are single use, cannot nest, and remain separate across concurrent async tasks. `parse_replay_recording()`, `replay_recording_from_case()`, `replay_model()` and `LocalRunApi` are also public APIs.

The runtime verifies bounded JSON and uses shared TypeScript fixtures for floating-point numbers, negative zero, Unicode ordering, boolean/number distinction and raw capture digests. No SDK publication, live provider test or production traffic is implied by these local proofs.
