"""Execute a native child agent with bounded, private replay files."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import yaml

from .contracts import (
    MAX_RECORDING_BYTES,
    ReplayError,
    object_fields,
    parse_json_bytes,
    parse_replay_recording,
)
from .environment import bounded_file, write_private_json
from .provider import ecmascript_json
from .session import ReplaySession


def parse_agent_config(content):
    if len(content.encode()) > 64 * 1024:
        raise ReplayError("Agent config exceeds 64 KiB.")

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        pairs = loader.construct_pairs(node, deep=True)
        if any(type(key) is not str for key, _ in pairs) or len(dict(pairs)) != len(pairs):
            raise ReplayError("Agent config contains duplicate or invalid keys.")
        return dict(pairs)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        depth = nodes = 0
        for event in yaml.parse(content):
            nodes += 1
            if isinstance(event, yaml.events.AliasEvent) or nodes > 4096:
                raise ReplayError("Agent config aliases and oversized structures are unsupported.")
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                depth += 1
            if isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                depth -= 1
            if depth > 32:
                raise ReplayError("Agent config is nested too deeply.")
        value = object_fields(
            yaml.load(content, Loader=Loader), {"command"}, {"input_from_trace", "timeout_seconds"}
        )
    except (yaml.YAMLError, UnicodeError, TypeError, RecursionError):
        raise ReplayError("Agent config must contain bounded valid YAML.") from None
    command = value["command"]
    if (
        type(command) is not list
        or not 1 <= len(command) <= 100
        or any(
            type(part) is not str or not 1 <= len(part) <= 4096 or "\x00" in part
            for part in command
        )
    ):
        raise ReplayError("Agent command must contain 1 to 100 argument strings.")
    result = {"command": command}
    if "input_from_trace" in value:
        mapping = value["input_from_trace"]
        if type(mapping) is not dict or any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key)
            or type(path) is not str
            or not 1 <= len(path) <= 200
            for key, path in mapping.items()
        ):
            raise ReplayError("Agent input mapping is invalid.")
        result["input_from_trace"] = mapping
    timeout = value.get("timeout_seconds", 300)
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ReplayError("Agent timeout must be from 1 to 3600 seconds.")
    result["timeout_seconds"] = timeout
    return result


def _mapped_input(value, mapping):
    if mapping is None:
        return value
    result = {}
    for key, path in mapping.items():
        current = value
        for part in path.split("."):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", part) or part in {
                "__proto__",
                "prototype",
                "constructor",
            }:
                raise ReplayError("Agent input mapping does not resolve.")
            if type(current) is dict and part in current:
                current = current[part]
            elif type(current) is list and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                raise ReplayError("Agent input mapping does not resolve.")
        result[key] = current
    return result


def parse_report(value):
    data = object_fields(
        value,
        {
            "sourceTraceId",
            "mode",
            "hits",
            "liveCalls",
            "switchedToLive",
            "reason",
            "firstMiss",
            "traceIds",
        },
    )
    if (
        type(data["sourceTraceId"]) is not str
        or not 1 <= len(data["sourceTraceId"]) <= 200
        or data["mode"] not in {"tools_only", "until_span"}
        or any(type(data[key]) is not int or data[key] < 0 for key in ("hits", "liveCalls"))
        or type(data["switchedToLive"]) is not bool
        or data["reason"] not in {None, "checkpoint", "missing_recording"}
    ):
        raise ReplayError("Invalid local replay activity report.")
    miss = data["firstMiss"]
    if miss is not None:
        object_fields(miss, {"kind", "name"})
        if (
            miss["kind"] not in {"tool", "model"}
            or type(miss["name"]) is not str
            or len(miss["name"]) > 200
        ):
            raise ReplayError("Invalid replay miss report.")
    traces = data["traceIds"]
    if (
        type(traces) is not list
        or len(traces) > 2
        or any(
            type(trace) is not str or not re.fullmatch(r"[a-f0-9]{32}", trace) for trace in traces
        )
    ):
        raise ReplayError("Invalid replay trace report.")
    return data


def _execute(command, input, *, cwd, env, timeout):
    child = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    events = queue.Queue(maxsize=16)
    stop = threading.Event()

    def emit(event):
        while not stop.is_set():
            try:
                events.put(event, timeout=0.1)
                return
            except queue.Full:
                pass

    def read(kind, stream):
        try:
            while not stop.is_set():
                chunk = stream.read(65536)
                if not chunk:
                    break
                emit((kind, chunk))
        finally:
            emit((kind, None))

    def write():
        try:
            child.stdin.write(input)
            child.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    workers = [
        threading.Thread(target=read, args=(kind, stream), daemon=True)
        for kind, stream in (("stdout", child.stdout), ("stderr", child.stderr))
    ]
    workers.append(threading.Thread(target=write, daemon=True))
    for worker in workers:
        worker.start()
    chunks = {"stdout": [], "stderr": []}
    bytes_read = ended = 0
    deadline = time.monotonic() + timeout
    completed = False
    try:
        while ended < 2 or child.poll() is None:
            if time.monotonic() >= deadline:
                raise ReplayError("Local agent exceeded its time limit.")
            try:
                kind, chunk = events.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if chunk is None:
                ended += 1
                continue
            bytes_read += len(chunk)
            if bytes_read > MAX_RECORDING_BYTES:
                raise ReplayError("Local agent output exceeds 16 MiB.")
            chunks[kind].append(chunk)
        completed = True
        return {
            "exitCode": child.returncode,
            **{
                key: b"".join(parts).decode("utf-8", errors="replace")
                for key, parts in chunks.items()
            },
        }
    finally:
        stop.set()
        if not completed:
            try:
                os.killpg(child.pid, signal.SIGKILL) if os.name != "nt" else child.kill()
            except ProcessLookupError:
                pass
            if child.poll() is None:
                child.wait(timeout=5)
        for worker in workers:
            worker.join(timeout=1)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()


def run_local_command(
    *,
    recording,
    replay,
    input,
    config,
    cwd,
    input_context=None,
    env=None,
    label="local-replay",
    params=None,
):
    saved = parse_replay_recording(recording)
    ReplaySession(
        saved, mode=replay["mode"], until_span=replay.get("untilSpan"), run=replay.get("run")
    )
    mapped = _mapped_input(
        input_context if "input_from_trace" in config else input, config.get("input_from_trace")
    )
    data = ecmascript_json(mapped) + b"\n"
    if len(data) > MAX_RECORDING_BYTES:
        raise ReplayError("Agent input exceeds 16 MiB.")
    with tempfile.TemporaryDirectory(prefix="neatlogs-replay-") as directory:
        file, report = Path(directory) / "recording.json", Path(directory) / "report.json"
        write_private_json(file, saved.to_dict(), limit=MAX_RECORDING_BYTES)
        binding = replay.get("run") or {}
        environment = dict(os.environ if env is None else env)
        environment.update(
            {
                "NEATLOGS_REPLAY_FILE": str(file),
                "NEATLOGS_REPLAY_REPORT": str(report),
                "NEATLOGS_REPLAY_TRACE_ID": saved.source_trace_id,
                "NEATLOGS_REPLAY_MODE": replay["mode"],
                "NEATLOGS_REPLAY_UNTIL": replay.get("untilSpan", ""),
                "NEATLOGS_REPLAY_LABEL": label,
                "NEATLOGS_REPLAY_RUN_ID": binding.get("runId", ""),
                "NEATLOGS_REPLAY_CASE_ID": binding.get("caseId", ""),
                "NEATLOGS_REPLAY_TRIAL": str(binding.get("trial", "")),
                "NEATLOGS_REPLAY_PARAMS": ecmascript_json(params or {}).decode(),
            }
        )
        try:
            result = _execute(
                config["command"],
                data,
                cwd=str(Path(cwd).absolute()),
                env=environment,
                timeout=config["timeout_seconds"],
            )
        except OSError:
            raise ReplayError("Local agent command could not be started.") from None
        try:
            status = parse_report(
                parse_json_bytes(bounded_file(report, 16 * 1024), limit=16 * 1024)
            )
        except ReplayError:
            raise ReplayError(
                "The agent did not report replay activity. Call init() in the child process."
            ) from None
        if status["sourceTraceId"] != saved.source_trace_id or status["mode"] != replay["mode"]:
            raise ReplayError("Local replay report does not match the requested recording.")
        if status["hits"] + status["liveCalls"] == 0:
            raise ReplayError(
                "No replay-aware calls ran. Mark JSON tools with replay=True or use replay_model()."
            )
        return {**result, "status": status}
