"""Isolated dependency and instrumentation smoke checks for published versions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

LIBRARY_ALIASES = {
    "azure-ai-inference": "azure_ai_inference",
    "qdrant-langchain": "qdrant",
    "llama-index": "llamaindex",
    "vertex-google-genai": "vertex_ai",
    "legacy-vertex-ai": "vertexai",
    "autogen-agentchat": "autogen",
}

PROBE = """
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

package = os.environ["COMPAT_PACKAGE"]
expected_version = os.environ["COMPAT_VERSION"]
library = os.environ["COMPAT_LIBRARY"]
actual_version = importlib.metadata.version(package)
assert actual_version == expected_version, f"{package}: expected {expected_version}, got {actual_version}"

import neatlogs
assert Path(neatlogs.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), (
    f"Neatlogs was imported from the checkout instead of the isolated environment: {neatlogs.__file__}"
)
neatlogs.init(
    api_key="compatibility-only",
    instrumentations=[library],
    disable_export=True,
    register_shutdown_handlers=False,
)
try:
    init_module = importlib.import_module("neatlogs.init")
    manager = init_module._instrumentation_manager
    assert manager is not None and library in manager.instrumented, (
        f"{library} instrumentation did not activate"
    )
    tracer = manager.provider.get_tracer("neatlogs.compatibility")
    tracer.start_span("compatibility-activation-smoke").end()
finally:
    assert neatlogs.shutdown(timeout_millis=1000)
print(f"{package} {actual_version}: {library} activation smoke passed")
"""


def instrumented_library(integration: str) -> str:
    return LIBRARY_ALIASES.get(integration, integration.replace("-", "_"))


def run_command(
    command: list[str], *, timeout: int, env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            env=env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "exitCode": result.returncode,
            "output": (result.stdout + "\n" + result.stderr)[-3000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "timedOut": True, "output": f"Timed out after {timeout}s"}


def check_version(
    *, package: str, version: str | None, integration: str, extra: str, wheel: Path
) -> dict[str, Any]:
    if not version:
        return {"status": "not-tested", "stage": "version", "reason": "No recorded baseline version"}
    isolated_env = os.environ.copy()
    isolated_env.pop("PYTHONPATH", None)
    isolated_env.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="neatlogs-compat-") as temporary:
        environment = Path(temporary) / ".venv"
        created = run_command(
            [sys.executable, "-m", "venv", str(environment)], timeout=60,
            env=isolated_env,
        )
        if not created["ok"]:
            return {"status": "not-tested", "stage": "environment", "reason": created["output"]}
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        pip_env = {**isolated_env, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}
        installed = run_command(
            [
                str(python), "-m", "pip", "install", "--progress-bar", "off",
                "--retries", "1", "--timeout", "30",
                f"{wheel.resolve()}[{extra}]", f"{package}=={version}",
            ],
            timeout=240,
            env=pip_env,
        )
        if not installed["ok"]:
            return {"status": "blocked", "stage": "install", "reason": installed["output"]}
        checked = run_command(
            [str(python), "-m", "pip", "check"], timeout=30,
            env=isolated_env,
        )
        if not checked["ok"]:
            return {"status": "fail", "stage": "dependency-check", "reason": checked["output"]}
        probe_env = isolated_env.copy()
        probe_env.update(
            {
                "COMPAT_PACKAGE": package,
                "COMPAT_VERSION": version,
                "COMPAT_LIBRARY": instrumented_library(integration),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
        )
        probed = run_command(
            [str(python), "-c", PROBE], timeout=60, env=probe_env,
            cwd=environment,
        )
        if not probed["ok"]:
            return {"status": "fail", "stage": "instrumentation-activation", "reason": probed["output"]}
        return {"status": "pass", "stage": "instrumentation-activation", "reason": probed["output"].strip()}


def compare_results(baseline: dict[str, Any], latest: dict[str, Any]) -> str:
    if baseline["status"] == "pass" and latest["status"] == "fail":
        return "smoke-regression"
    if latest["status"] == "pass":
        return "latest-smoke-passed"
    if latest["status"] == "fail":
        return "latest-failure-needs-triage"
    if latest["status"] == "blocked":
        return "latest-install-blocked"
    return "not-tested"


def run_pair(
    *, package: str, baseline_version: str | None, latest_version: str,
    integration: str, extra: str, wheel: Path,
) -> dict[str, Any]:
    baseline = check_version(
        package=package, version=baseline_version, integration=integration,
        extra=extra, wheel=wheel,
    )
    latest = check_version(
        package=package, version=latest_version, integration=integration,
        extra=extra, wheel=wheel,
    )
    return {
        "package": package,
        "integration": integration,
        "extra": extra,
        "baselineVersion": baseline_version,
        "latestVersion": latest_version,
        "baseline": baseline,
        "latest": latest,
        "comparison": compare_results(baseline, latest),
        "scope": "Exact-version install, dependency check, and Neatlogs instrumentation activation; no provider call",
    }


def not_tested_pair(
    package: str, baseline_version: str | None, latest_version: str,
    integration: str, reason: str,
) -> dict[str, Any]:
    outcome = {"status": "not-tested", "stage": "setup", "reason": reason}
    return {
        "package": package,
        "integration": integration,
        "baselineVersion": baseline_version,
        "latestVersion": latest_version,
        "baseline": outcome,
        "latest": outcome,
        "comparison": "not-tested",
        "scope": "Exact-version install, dependency check, and instrumentation activation",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--latest", required=True)
    parser.add_argument("--integration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((Path(__file__).resolve().parents[2] / ".compatibility/integrations.json").read_text())
    integrations = {item["id"]: item for item in config["integrations"]}
    if args.integration not in integrations or args.package not in integrations[args.integration]["packages"]:
        raise ValueError("The requested package/integration pair is not in the monitoring inventory")
    extra = integrations[args.integration].get("extra")
    wheels = list(Path("dist").glob("neatlogs-*.whl"))
    if not extra or len(wheels) != 1:
        reason = (
            f"No installation extra configured for {args.integration}"
            if not extra
            else f"Expected one SDK wheel, found {len(wheels)}"
        )
        result = not_tested_pair(
            args.package, args.baseline, args.latest, args.integration, reason
        )
    else:
        result = run_pair(
            package=args.package,
            baseline_version=args.baseline,
            latest_version=args.latest,
            integration=args.integration,
            extra=extra,
            wheel=wheels[0],
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{json.dumps(result, indent=2)}\n")
    print(f"{args.package}/{args.integration}: {result['comparison']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
