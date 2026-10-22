from google.genai import types

from neatlogs.core.choice_accumulator import ChoiceAccumulator, _finish_reason


def test_finish_reason_uses_enum_value():
    assert _finish_reason(types.FinishReason.STOP) == "STOP"
    assert _finish_reason(types.FinishReason.MAX_TOKENS) == "MAX_TOKENS"


def test_finish_reason_keeps_plain_strings():
    assert _finish_reason("stop") == "stop"


def test_google_candidate_finish_reason_is_wire_value():
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(role="model", parts=[types.Part(text="hi")]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    acc = ChoiceAccumulator()
    acc.add_google_response(response)
    assert acc.choices[0].finish_reason == "STOP"
