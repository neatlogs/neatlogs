from __future__ import annotations

from pathlib import Path

from neatlogs.prompt.cli import main
from neatlogs.prompt.sync import (
    DiscoveredPrompt,
    discover_templates,
    normalize_prompt_name,
    plan_prompt_sync,
    push_templates,
)


def _write_prompt_module(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from neatlogs import SystemPromptTemplate, UserPromptTemplate",
                "",
                "SUPPORT_DRAFTER_SYSTEM_V4 = SystemPromptTemplate(",
                "    'You are a support agent for {{company}}.'",
                ")",
                "",
                "SUPPORT_DRAFTER_USER = UserPromptTemplate([",
                "    {'role': 'user', 'content': 'Question: {{question}}'}",
                "])",
                "",
                "IGNORED = 'not a prompt'",
            ]
        )
    )


def test_normalize_prompt_name_converts_constants_to_kebab_case():
    assert normalize_prompt_name("SUPPORT_DRAFTER_USER") == "support-drafter-user"
    assert (
        normalize_prompt_name("SUPPORT_DRAFTER_SYSTEM_V4", prefix="support-copilot")
        == "support-copilot-support-drafter-system-v4"
    )


def test_discover_templates_from_python_module(tmp_path):
    module_path = tmp_path / "prompts.py"
    _write_prompt_module(module_path)

    prompts = discover_templates(module_path)

    assert [prompt.name for prompt in prompts] == [
        "support-drafter-system-v4",
        "support-drafter-user",
    ]
    assert prompts[0].type == "text"
    assert prompts[1].type == "chat"


def test_plan_prompt_sync_marks_create_version_and_unchanged():
    prompts = [
        DiscoveredPrompt("NEW", "new", "hello", "text"),
        DiscoveredPrompt("CHANGED", "changed", "new text", "text"),
        DiscoveredPrompt("SAME", "same", "same text", "text"),
    ]

    results = plan_prompt_sync(
        prompts,
        {
            "new": None,
            "changed": {"type": "text", "content": "old text"},
            "same": {"type": "text", "content": "same text"},
        },
    )

    assert [result.action for result in results] == ["create", "version", "unchanged"]


def test_push_templates_uses_create_version_and_skip(tmp_path):
    module_path = tmp_path / "prompts.py"
    _write_prompt_module(module_path)
    client = FakePromptClient(
        {
            "support-drafter-system-v4": None,
            "support-drafter-user": {
                "items": [
                    {
                        "version": 1,
                        "type": "chat",
                        "messages": [{"role": "user", "content": "Old"}],
                    }
                ]
            },
        }
    )

    results = push_templates(module_path, label="production", client=client)

    assert [result.action for result in results] == ["create", "version"]
    assert client.created[0]["name"] == "support-drafter-system-v4"
    assert client.versioned[0]["prompt_name"] == "support-drafter-user"


def test_push_templates_dry_run_does_not_write(tmp_path):
    module_path = tmp_path / "prompts.py"
    _write_prompt_module(module_path)
    client = FakePromptClient({"support-drafter-system-v4": None, "support-drafter-user": None})

    results = push_templates(module_path, label="production", client=client, dry_run=True)

    assert [result.action for result in results] == ["create", "create"]
    assert client.created == []
    assert client.versioned == []


def test_cli_dry_run_prints_discovered_templates(tmp_path, capsys):
    module_path = tmp_path / "prompts.py"
    _write_prompt_module(module_path)

    exit_code = main(["push", str(module_path), "--label", "production", "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "discovered" in output
    assert "support-drafter-system-v4" in output
    assert "support-drafter-user" in output


class FakePromptClient:
    def __init__(self, prompts):
        self.prompts = prompts
        self.created = []
        self.versioned = []

    def list_prompts(self, *, name):
        value = self.prompts.get(name)
        if value is None:
            return {"items": []}
        return value

    def create_prompt(self, **kwargs):
        self.created.append(kwargs)
        return object()

    def save_as_version(self, **kwargs):
        self.versioned.append(kwargs)
        return {}
