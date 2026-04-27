"""
Version helpers for the Neatlogs SDK.
"""

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path


def _version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    match = re.search(
        r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$',
        pyproject_path.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def get_version() -> str:
    try:
        return package_version("neatlogs")
    except PackageNotFoundError:
        return _version_from_pyproject() or "0.0.0"


__version__ = get_version()
