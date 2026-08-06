import importlib.util
from pathlib import Path

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "sdk_examples" / "crewai_prompt_fidelity.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("crewai_prompt_fidelity_example", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_builder_places_configured_tail_after_100000() -> None:
    example = _load_example()

    description = example.build_long_task_description(target_tail_offset=100_000)

    assert description.index(example.NEAR_10000_MARKER) < 10_000
    assert description.index(example.PROMPT_TARGET_TAIL_MARKER) >= 100_000


def test_tool_builder_returns_plain_string_with_tail_after_10000() -> None:
    example = _load_example()

    output = example.build_tool_fidelity_output(target_chars=12_500)

    assert isinstance(output, str)
    assert output.index(example.TOOL_NEAR_10000_MARKER) < 10_000
    assert output.index(example.TOOL_TAIL_AFTER_10000_MARKER) > 10_000
    assert len(output) >= 12_500


def test_default_prompt_probe_remains_bounded() -> None:
    example = _load_example()

    description = example.build_long_task_description()

    assert 12_000 < len(description) < 20_000
    assert example.PROMPT_TARGET_TAIL_MARKER in description
