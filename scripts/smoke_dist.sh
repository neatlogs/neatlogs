#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 path/to/neatlogs.whl" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
smoke_dir="$(mktemp -d)"
trap 'rm -rf "$smoke_dir"' EXIT

python_bin="${PYTHON:-python3}"
"$python_bin" -m venv "$smoke_dir/venv"
"$smoke_dir/venv/bin/python" -m pip install "$1"
cd "$smoke_dir"
"$smoke_dir/venv/bin/python" "$repo_root/scripts/smoke_readme.py"
"$smoke_dir/venv/bin/neatlogs-doctor" --disable-export --json > "$smoke_dir/doctor.json"
"$smoke_dir/venv/bin/python" -c \
  'import json, pathlib, sys; result=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert result["format_version"] == "neatlogs.doctor/v1" and result["ready"]' \
  "$smoke_dir/doctor.json"
