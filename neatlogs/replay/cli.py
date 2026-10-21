"""Native Python local simulation CLI. Agent execution never delegates to Node."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .api import LocalRunApi
from .contracts import (
    MAX_RECORDING_BYTES,
    ReplayError,
    json_value,
    parse_json_bytes,
    parse_replay_recording,
    uuid,
)
from .environment import bounded_file
from .frozen_case import replay_recording_from_case
from .local_command import parse_agent_config, run_local_command
from .session import ReplaySession


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ReplayError("Invalid local simulation arguments. Use --help for supported options.")


def _integer(value, lower, upper):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ReplayError("Invalid local replay integer option.") from None
    if not lower <= result <= upper:
        raise ReplayError(f"Local replay integer must be from {lower} to {upper}.")
    return result


def _parameters(values):
    if len(values) > 20:
        raise ReplayError("Use at most 20 parameters.")
    result = {}
    for raw in values:
        name, separator, text = raw.partition("=")
        if not separator or not 1 <= len(name) <= 100 or len(text) > 2000 or name in result:
            raise ReplayError(
                "Use unique --param name=value options, up to 2000 characters per value."
            )
        try:
            value = json.loads(text)
        except ValueError:
            value = text
        if value is not None and type(value) not in (str, int, float, bool):
            raise ReplayError("Parameters must be strings, finite numbers, booleans or null.")
        result[name] = json_value(value)
    return result


def _api():
    return LocalRunApi(
        project_id=os.getenv("NEATLOGS_PROJECT_ID", ""),
        token=os.getenv("NEATLOGS_TOKEN", ""),
        base_url=os.getenv("NEATLOGS_HOST", "https://app.neatlogs.com"),
    )


def _remaining(run):
    if run is None:
        return float("inf")
    return (
        datetime.fromisoformat(run["deadlineAt"].replace("Z", "+00:00"))
        - datetime.now(timezone.utc)
    ).total_seconds()


def _parser():
    parser = Parser(
        prog="neatlogs simulate local",
        description="Run your local Python agent with saved tool results. Models remain live in tools_only mode. The first missing match switches all subsequent calls live.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recording")
    source.add_argument("--dataset")
    source.add_argument("--resume")
    parser.add_argument("--version")
    parser.add_argument("--case", action="append", default=[], dest="cases")
    parser.add_argument("--agent", default=".neatlogs/agent.yaml")
    parser.add_argument("--mode", choices=["tools_only", "until_span"])
    parser.add_argument("--until-span")
    parser.add_argument("--label", default="Local replay")
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--trials", default="1")
    parser.add_argument("--deadline-seconds", default="900")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--form")
    parser.add_argument("--evaluator", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--state")
    parser.add_argument("--batch-size", default=None)
    return parser


def run_simulate_cli(argv):
    run = None
    try:
        if argv[:2] != ["simulate", "local"]:
            raise ReplayError("Use neatlogs simulate local --help.")
        args = _parser().parse_args(argv[2:])
        if args.resume:
            from .all_command import resume_whole_dataset

            if any(
                token.startswith("--") and token.split("=")[0] not in {"--resume", "--json"}
                for token in argv[2:]
            ):
                raise ReplayError("Resume accepts --resume and optional --json only.")
            return resume_whole_dataset(args.resume, api=_api(), as_json=args.json)
        if args.recording and (args.version or args.cases or args.upload or args.all):
            raise ReplayError("Recording files cannot use dataset version, case or upload options.")
        if args.all and (not args.dataset or args.cases or not args.state):
            raise ReplayError("Use --all with --dataset, --version and a new --state file.")
        if not args.all and (args.state or args.batch_size):
            raise ReplayError("--state and --batch-size require --all.")
        if not args.upload and (args.form or args.evaluator):
            raise ReplayError("Use --upload to apply grading settings.")
        mode = args.mode or ("until_span" if args.until_span else "tools_only")
        if (mode == "tools_only" and args.until_span) or (
            mode == "until_span" and not args.until_span
        ):
            raise ReplayError("Use tools_only, or supply --until-span for checkpoint replay.")
        replay = {"mode": mode, **({"untilSpan": args.until_span} if args.until_span else {})}
        trials = _integer(args.trials, 1, 10)
        deadline = _integer(args.deadline_seconds, 60, 3600)
        params = _parameters(args.param)
        label = args.label.strip()
        if not 1 <= len(label) <= 120:
            raise ReplayError("Use a label from 1 to 120 characters.")
        config = parse_agent_config(bounded_file(args.agent, 64 * 1024).decode())
        dataset, version = (
            (uuid(args.dataset), uuid(args.version)) if args.dataset else (None, None)
        )
        cases = [uuid(case) for case in args.cases]
        if (
            dataset
            and not args.all
            and (not 1 <= len(cases) <= 100 or len(set(cases)) != len(cases))
        ):
            raise ReplayError("Select 1 to 100 different examples with --case.")
        grading = None
        if args.upload:
            evaluators = [uuid(evaluator) for evaluator in args.evaluator]
            if not 1 <= len(evaluators) <= 20 or len(set(evaluators)) != len(evaluators):
                raise ReplayError("Choose 1 to 20 different evaluators.")
            grading = {"formVersionId": uuid(args.form), "evaluatorIds": evaluators}
        api = _api() if dataset else None
        if args.all:
            from .all_command import start_whole_dataset

            return start_whole_dataset(
                args.state,
                api=api,
                dataset_id=dataset,
                version_id=version,
                config=config,
                agent_file=args.agent,
                replay=replay,
                label=label,
                params=params,
                trials=trials,
                deadline_seconds=deadline,
                grading=grading,
                batch_size=_integer(args.batch_size or 25, 1, 100),
                as_json=args.json,
            )
        sources, size = [], 0
        for case_id in cases if dataset else [None]:
            item = (
                api.frozen_case(dataset, version, case_id).get("case")
                if case_id
                else parse_json_bytes(bounded_file(args.recording, MAX_RECORDING_BYTES))
            )
            if type(item) is not dict or case_id and item.get("caseId") != case_id:
                raise ReplayError("The saved source does not match the requested example.")
            if case_id:
                uuid(item.get("revisionId"))
            size += len(json.dumps(item).encode())
            if size > 32 * 1024 * 1024:
                raise ReplayError("Selected recordings exceed 32 MiB. Select fewer examples.")
            converted = (
                {"recording": parse_replay_recording(item), "skippedTools": 0}
                if "entries" in item
                else replay_recording_from_case(item)
            )
            ReplaySession(converted["recording"], mode=mode, until_span=args.until_span)
            sources.append((case_id, item, converted))
        if grading:
            run = api.register(
                {
                    "datasetVersionId": version,
                    "label": label,
                    "caseIds": cases,
                    "trials": trials,
                    "params": params,
                    "grading": grading,
                    "deadlineSeconds": deadline,
                    "replay": replay,
                }
            )
        reports = []
        output_bytes = 0
        for case_id, item, converted in sources:
            for trial in range(1, trials + 1):
                report = {
                    "caseId": case_id,
                    "trial": trial,
                    "skippedTools": converted["skippedTools"],
                    "uploaded": False,
                }
                reports.append(report)
                try:
                    remaining = _remaining(run)
                    if remaining < 1 or output_bytes > 32 * 1024 * 1024:
                        raise ReplayError(
                            "Run deadline or output limit reached. This example was not executed."
                        )
                    settings = {
                        **replay,
                        **(
                            {"run": {"runId": run["runId"], "caseId": case_id, "trial": trial}}
                            if run
                            else {}
                        ),
                    }
                    timeout = (
                        config["timeout_seconds"]
                        if run is None
                        else min(config["timeout_seconds"], max(1, int(remaining)))
                    )
                    result = run_local_command(
                        recording=converted["recording"],
                        replay=settings,
                        input=item.get("input"),
                        input_context={
                            **item,
                            "root": {
                                "input": item.get("input"),
                                "output": item.get("observedOutput"),
                            },
                        },
                        config={**config, "timeout_seconds": timeout},
                        cwd=os.getcwd(),
                        label=label,
                        params=params,
                    )
                    report["result"] = result
                    output_bytes += len(result["stdout"].encode()) + len(result["stderr"].encode())
                    if result["exitCode"] != 0:
                        raise ReplayError(
                            "The agent exited unsuccessfully; no output was uploaded."
                        )
                    if output_bytes > 32 * 1024 * 1024:
                        raise ReplayError("Combined agent output exceeds 32 MiB.")
                    if run:
                        if len(result["status"]["traceIds"]) != 1:
                            raise ReplayError(
                                "Upload requires exactly one workflow trace. Wrap the agent in one root span and call shutdown()."
                            )
                        receipt = api.upload(
                            run["runId"],
                            case_id=case_id,
                            trial=trial,
                            trace_id=result["status"]["traceIds"][0],
                            wait_millis=max(1, min(60_000, int(_remaining(run) * 1000))),
                        )
                        report.update(uploaded=True, outputId=receipt["outputId"])
                except (ReplayError, OSError, UnicodeError) as error:
                    report["error"] = (
                        str(error)
                        if isinstance(error, ReplayError)
                        else "Local agent execution failed."
                    )
                if not args.json:
                    if "result" in report:
                        print(report["result"]["stdout"], end="")
                        print(report["result"]["stderr"], end="", file=sys.stderr)
                    print(
                        f"{case_id or 'Recording'}, attempt {trial}: {report.get('error') or ('Saved to Runs; grading is pending.' if report['uploaded'] else 'Finished locally.')} "
                    )
        exit_code = int(any("error" in report for report in reports))
        if args.json:
            output = {
                "exitCode": exit_code,
                "runId": run["runId"] if run else None,
                "reports": reports,
            }
            if len(reports) == 1:
                first = reports[0]
                output = {
                    **first.get("result", {}),
                    **{key: value for key, value in first.items() if key != "result"},
                    "exitCode": exit_code,
                    "runId": output["runId"],
                }
            print(json.dumps(output, ensure_ascii=False))
        return exit_code
    except (ReplayError, ValueError, OSError, UnicodeError) as error:
        message = (
            str(error) if isinstance(error, ReplayError) else "Local replay configuration failed."
        )
        print(message + (f" Run: {run['runId']}." if run else ""), file=sys.stderr)
        return 1
