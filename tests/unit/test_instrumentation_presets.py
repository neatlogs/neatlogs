from neatlogs.init import _resolve_instrumentations


def test_resolve_instrumentations_keeps_explicit_libraries():
    assert _resolve_instrumentations(["openai"]) == ([], ["openai"])


def test_resolve_instrumentations_extracts_preset_tags():
    assert _resolve_instrumentations(["llm", "agent"]) == (["llm", "agent"], [])


def test_resolve_instrumentations_splits_mixed_entries():
    assert _resolve_instrumentations(["llm", "openai"]) == (["llm"], ["openai"])


def test_resolve_instrumentations_keeps_unknown_entries_as_libraries():
    assert _resolve_instrumentations(["custom_sdk"]) == ([], ["custom_sdk"])
