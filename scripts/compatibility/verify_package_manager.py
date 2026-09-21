from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

VERIFY_PROGRAM = """
import neatlogs

assert callable(neatlogs.init)
assert callable(neatlogs.trace)
assert callable(neatlogs.shutdown)
print("Installed consumer verified")
"""


def run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def virtualenv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def built_wheel() -> Path:
    wheels = sorted((REPOSITORY_ROOT / "dist").glob("neatlogs-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected one built wheel, found {len(wheels)}")
    return wheels[0].resolve()


def verify_consumer(manager: str) -> None:
    if manager not in {"pip", "uv", "poetry"}:
        raise ValueError(f"unsupported package manager: {manager}")
    wheel = built_wheel()
    with tempfile.TemporaryDirectory(prefix=f"neatlogs-{manager}-consumer-") as temporary:
        directory = Path(temporary)
        verify_path = directory / "verify.py"
        verify_path.write_text(VERIFY_PROGRAM)

        if manager == "pip":
            environment = directory / ".venv"
            run(sys.executable, "-m", "venv", str(environment))
            python = virtualenv_python(environment)
            run(str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel))
            run(str(python), str(verify_path))
        elif manager == "uv":
            environment = directory / ".venv"
            run("uv", "venv", "--python", sys.executable, str(environment))
            python = virtualenv_python(environment)
            run("uv", "pip", "install", "--python", str(python), str(wheel))
            run(str(python), str(verify_path))
        else:
            run(
                "poetry",
                "init",
                "--name",
                "neatlogs-compatibility-consumer",
                "--python",
                ">=3.10,<3.14",
                "--no-interaction",
                cwd=directory,
            )
            run("poetry", "add", str(wheel), "--no-interaction", cwd=directory)
            run("poetry", "run", "python", str(verify_path), cwd=directory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manager", choices=["pip", "uv", "poetry"], required=True)
    args = parser.parse_args()
    verify_consumer(args.manager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
