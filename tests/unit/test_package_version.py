import re
from pathlib import Path

import neatlogs


def test_package_version_matches_pyproject():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    match = re.search(
        r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$',
        pyproject_path.read_text(encoding="utf-8"),
    )

    assert match is not None
    assert neatlogs.__version__ == match.group(1)
