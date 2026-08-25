"""Offline smoke of the README's install/init/decorate/flush lifecycle."""

import neatlogs
from neatlogs import span

doctor = neatlogs.doctor(disable_export=True)
assert doctor.ready
assert doctor.format_version == "neatlogs.doctor/v1"
assert neatlogs.init.__module__ == "neatlogs.init"

neatlogs.init(
    api_key="readme-smoke-key",
    workflow_name="readme-smoke",
    instrumentations=[],
    disable_export=True,
)


@span(kind="WORKFLOW", name="quickstart")
def main():
    return "smoke-ok"


assert main() == "smoke-ok"
assert neatlogs.flush()
assert neatlogs.shutdown()
