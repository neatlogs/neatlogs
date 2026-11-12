from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from neatlogs_harbor import syncer as syncer_module
from neatlogs_harbor.client import NeatlogsClient

from .fake_neatlogs import INGEST_KEY, PROJECT_ID, TOKEN, FakeNeatlogs

FIXTURES = Path(__file__).parent / "fixtures" / "jobs"

# Secret-shaped strings are built at runtime so no real-looking token is committed.
FAKE_GITHUB_TOKEN = "ghp_" + "Ab1" * 12
FAKE_OPENAI_KEY = "sk-proj-" + "Zy9" * 10


@pytest.fixture
def fake() -> FakeNeatlogs:
    return FakeNeatlogs()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    values = {
        "NEATLOGS_API_KEY": INGEST_KEY,
        "NEATLOGS_TOKEN": TOKEN,
        "NEATLOGS_PROJECT_ID": PROJECT_ID,
        "NEATLOGS_HOST": "https://neatlogs.test",
        "NEATLOGS_ENDPOINT": "https://ingest.neatlogs.test",
        "NEATLOGS_HARBOR_FORM_VERSION_ID": "0b7f2b9e-8a3c-4d1e-9f60-2a7c5e4d3b21",
        "NEATLOGS_HARBOR_EVALUATOR_IDS": "4e2d9c1a-7b3f-4a8e-b6d5-1c0f9e8d7a65",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.fixture
def wire(fake: FakeNeatlogs, monkeypatch: pytest.MonkeyPatch) -> FakeNeatlogs:
    """Route every client the integration builds to the fake server, without sleeping."""

    def build(**kwargs):
        kwargs.setdefault("transport", fake.transport())
        kwargs["sleep"] = lambda _seconds: None
        return NeatlogsClient(**kwargs)

    monkeypatch.setattr(syncer_module, "NeatlogsClient", build)
    return fake


def copy_job(tmp_path: Path, name: str) -> Path:
    target = tmp_path / "jobs" / name
    shutil.copytree(FIXTURES / name, target)
    for path in target.rglob("*.json"):
        text = path.read_text()
        if "{{" in text:
            text = text.replace("{{GITHUB_TOKEN}}", FAKE_GITHUB_TOKEN)
            text = text.replace("{{OPENAI_KEY}}", FAKE_OPENAI_KEY)
            path.write_text(text)
    return target


@pytest.fixture
def oracle_job(tmp_path: Path) -> Path:
    return copy_job(tmp_path, "oracle-hello-world")


@pytest.fixture
def mixed_job(tmp_path: Path) -> Path:
    return copy_job(tmp_path, "mixed-outcomes")
