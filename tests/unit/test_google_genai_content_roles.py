"""
Regression tests for google-genai input extraction.

A real multi-turn conversation passes history as typed ``types.Content`` objects
(role="user"/"model" + parts=[types.Part(...)]). The extractor previously handled
only str/dict items, so every typed turn was mislabeled "user" and its content
became a Python repr. These tests lock the fix: correct per-turn roles + clean
text, including tool (function_call / function_response) parts.
"""

import json

from neatlogs.google_genai import _normalize_content_item, _text_from_parts


# --- lightweight stand-ins for typed genai objects (no SDK needed) ----------


class _Part:
    def __init__(self, text=None, function_call=None, function_response=None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response


class _Content:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


class _FnCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FnResp:
    def __init__(self, name, response):
        self.name = name
        self.response = response


def test_plain_string_is_user():
    assert _normalize_content_item("hello") == ("user", "hello")


def test_dict_content_keeps_role_and_text():
    assert _normalize_content_item(
        {"role": "model", "parts": [{"text": "hi"}]}
    ) == ("model", "hi")


def test_typed_content_user_role_and_text():
    c = _Content("user", [_Part(text="What is 2+2?")])
    assert _normalize_content_item(c) == ("user", "What is 2+2?")


def test_typed_content_model_role_preserved():
    # The bug: a model/assistant turn used to be mislabeled "user".
    c = _Content("model", [_Part(text="4")])
    role, text = _normalize_content_item(c)
    assert role == "model"
    assert text == "4"


def test_typed_content_multi_part_text_joined():
    c = _Content("user", [_Part(text="line 1"), _Part(text="line 2")])
    assert _normalize_content_item(c) == ("user", "line 1\nline 2")


def test_typed_function_call_part_serialized():
    c = _Content("model", [_Part(function_call=_FnCall("get_price", {"plan": "pro"}))])
    role, text = _normalize_content_item(c)
    assert role == "model"
    payload = json.loads(text)
    assert payload["function_call"]["name"] == "get_price"
    assert payload["function_call"]["args"] == {"plan": "pro"}


def test_typed_function_response_part_serialized():
    c = _Content("user", [_Part(function_response=_FnResp("get_price", {"result": 199.99}))])
    role, text = _normalize_content_item(c)
    assert role == "user"
    payload = json.loads(text)
    assert payload["function_response"]["name"] == "get_price"
    assert payload["function_response"]["response"] == {"result": 199.99}


def test_text_from_parts_mixed():
    parts = [_Part(text="a"), {"text": "b"}, "c"]
    assert _text_from_parts(parts) == "a\nb\nc"
