"""
code.file.path is shortened to a home-relative path only when the file really is under the
home directory, compared by whole path components, and always uses forward slashes.
"""

from pathlib import Path, PurePosixPath
from unittest import mock

import pytest

from neatlogs.decorators import _base


def _target():
    pass


@pytest.mark.parametrize(
    "home, file, expected",
    [
        ("/home/al", "/home/al/app/agent.py", "app/agent.py"),
        # A sibling directory that merely starts with the home path is not under it.
        ("/home/al", "/home/alice/app/agent.py", "/home/alice/app/agent.py"),
        ("/home/al", "/srv/app/agent.py", "/srv/app/agent.py"),
    ],
)
def test_code_file_path_is_home_relative_only_under_home(home, file, expected):
    with (
        mock.patch.object(_base, "Path", PurePosixPath),
        mock.patch.object(PurePosixPath, "home", classmethod(lambda cls: cls(home)), create=True),
        mock.patch.object(_base.inspect, "getfile", return_value=file),
    ):
        attrs = _base._capture_code_attrs(_target)

    assert attrs["code.file.path"] == expected


def test_code_file_path_uses_forward_slashes():
    # Real filesystem: this test file lives somewhere; if it is under home the path is relative
    # and must not start with a separator of either kind.
    attrs = _base._capture_code_attrs(_target)
    path = attrs["code.file.path"]
    if Path(__file__).is_relative_to(Path.home()):
        assert not path.startswith(("/", "\\"))
        assert "\\" not in path
