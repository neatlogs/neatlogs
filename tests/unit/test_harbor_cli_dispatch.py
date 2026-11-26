import sys
import types

from neatlogs.__main__ import main


def test_harbor_command_forwards_to_the_harbor_package(monkeypatch):
    calls = []
    cli = types.ModuleType("neatlogs_harbor.cli")
    cli.main = lambda argv: calls.append(argv) or 0
    monkeypatch.setitem(sys.modules, "neatlogs_harbor", types.ModuleType("neatlogs_harbor"))
    monkeypatch.setitem(sys.modules, "neatlogs_harbor.cli", cli)

    assert main(["harbor", "import", "jobs/"]) == 0
    assert calls == [["import", "jobs/"]]


def test_harbor_command_explains_the_missing_package(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "neatlogs_harbor", None)

    assert main(["harbor", "sync", "jobs/latest"]) == 4
    assert "pip install neatlogs-harbor" in capsys.readouterr().err
