from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .client import PromptClient, _get_shared_client
from .template import SystemPromptTemplate, UserPromptTemplate

PromptValue = Union[str, List[Dict[str, str]]]


@dataclass(frozen=True)
class DiscoveredPrompt:
    attr_name: str
    name: str
    prompt: PromptValue
    type: str


@dataclass(frozen=True)
class PromptSyncResult:
    name: str
    action: str
    type: str
    detail: str = ""


def normalize_prompt_name(attr_name: str, prefix: Optional[str] = None) -> str:
    name = attr_name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = name.strip("-")
    if prefix:
        prefix_name = re.sub(r"[^a-z0-9]+", "-", prefix.strip().lower()).strip("-")
        if prefix_name:
            return f"{prefix_name}-{name}"
    return name


def discover_templates(
    module_path: Union[str, Path], prefix: Optional[str] = None
) -> List[DiscoveredPrompt]:
    path = Path(module_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prompt module not found: {path}")
    if path.suffix != ".py":
        raise ValueError("Prompt sync currently supports Python files only.")

    module_key = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem)
    module_name = f"_neatlogs_prompt_sync_{module_key}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import prompt module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass

    discovered: List[DiscoveredPrompt] = []
    for attr_name, value in vars(module).items():
        if attr_name.startswith("_"):
            continue
        if not isinstance(value, (SystemPromptTemplate, UserPromptTemplate)):
            continue
        prompt_type, prompt_value = _prompt_payload(value)
        discovered.append(
            DiscoveredPrompt(
                attr_name=attr_name,
                name=normalize_prompt_name(attr_name, prefix=prefix),
                prompt=prompt_value,
                type=prompt_type,
            )
        )

    return sorted(discovered, key=lambda item: item.name)


def plan_prompt_sync(
    prompts: Sequence[DiscoveredPrompt],
    existing: Mapping[str, Optional[Mapping[str, Any]]],
) -> List[PromptSyncResult]:
    results: List[PromptSyncResult] = []
    for prompt in prompts:
        current = existing.get(prompt.name)
        if current is None:
            results.append(PromptSyncResult(prompt.name, "create", prompt.type, "not found"))
            continue
        if _existing_matches(current, prompt):
            results.append(
                PromptSyncResult(prompt.name, "unchanged", prompt.type, "already current")
            )
        else:
            results.append(PromptSyncResult(prompt.name, "version", prompt.type, "content changed"))
    return results


def push_templates(
    module_path: Union[str, Path],
    *,
    label: str,
    prefix: Optional[str] = None,
    dry_run: bool = False,
    commit_message: Optional[str] = None,
    client: Optional[PromptClient] = None,
) -> List[PromptSyncResult]:
    if not label:
        raise ValueError("label is required")

    prompt_client = client or _get_shared_client()
    prompts = discover_templates(module_path, prefix=prefix)
    results: List[PromptSyncResult] = []

    for prompt in prompts:
        current = _latest_prompt(prompt_client.list_prompts(name=prompt.name))
        planned = plan_prompt_sync([prompt], {prompt.name: current})[0]
        if dry_run or planned.action == "unchanged":
            results.append(planned)
            continue

        try:
            if planned.action == "create":
                prompt_client.create_prompt(
                    name=prompt.name,
                    prompt=prompt.prompt,
                    type=prompt.type,
                    labels=[label],
                    commit_message=commit_message,
                )
            elif planned.action == "version":
                if prompt.type == "chat":
                    prompt_client.save_as_version(
                        prompt_name=prompt.name,
                        messages=prompt.prompt,  # type: ignore[arg-type]
                        labels=[label],
                        commit_message=commit_message,
                    )
                else:
                    prompt_client.save_as_version(
                        prompt_name=prompt.name,
                        content=prompt.prompt,  # type: ignore[arg-type]
                        labels=[label],
                        commit_message=commit_message,
                    )
        except Exception as exc:
            results.append(PromptSyncResult(prompt.name, "failed", prompt.type, str(exc)))
            continue

        results.append(planned)

    return results


def _prompt_payload(
    template: Union[SystemPromptTemplate, UserPromptTemplate],
) -> tuple[str, PromptValue]:
    value = template.template
    if isinstance(value, str):
        return "text", value
    return "chat", [dict(message) for message in value]


def _latest_prompt(listing: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    items = listing.get("items") or []
    if not items:
        return None
    return max(items, key=lambda item: item.get("version") or 0)


def _existing_matches(existing: Mapping[str, Any], prompt: DiscoveredPrompt) -> bool:
    existing_type = existing.get("type") or ("chat" if existing.get("messages") else "text")
    if existing_type != prompt.type:
        return False

    if prompt.type == "chat":
        return _normal_messages(existing.get("messages")) == _normal_messages(prompt.prompt)

    return str(existing.get("content") or "") == str(prompt.prompt)


def _normal_messages(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    messages: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, Mapping):
            messages.append({str(key): str(val) for key, val in item.items()})
    return messages
