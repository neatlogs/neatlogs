"""Check the installed 3.14 wheel without importing excluded adapters."""

import importlib.metadata
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import neatlogs

assert sys.version_info[:2] == (3, 14)
excluded = {
    "openinference-instrumentation-crewai",
    "openinference-instrumentation-instructor",
}
requirements = [Requirement(value) for value in importlib.metadata.requires("neatlogs")]
for name in excluded:
    requirement = next(r for r in requirements if canonicalize_name(r.name) == name)
    assert requirement.marker is not None
    assert not requirement.marker.evaluate({"python_version": "3.14", "extra": ""})
    assert requirement.marker.evaluate({"python_version": "3.13", "extra": ""})

neatlogs.init(
    api_key="python314-smoke",
    disable_export=True,
    instrumentations=[],
    register_shutdown_handlers=False,
)
assert neatlogs.shutdown(timeout_millis=1000)
print("Python 3.14 install/import/init and excluded-adapter markers pass")
