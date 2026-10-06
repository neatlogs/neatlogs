"""Regression test for #105: EVALUATOR spans must not be downgraded to UNKNOWN.

Python emits EVALUATOR (span_kind.py valid set; mapping.infer_span_kind_from_name
returns it for evaluate/score/metric names), but telemetry_v2._KINDS omitted it,
so _kind() fell through to UNKNOWN. The bundled schema enum omitted it too.
"""

import json
from pathlib import Path

from neatlogs.core.telemetry_v2 import _KINDS
from neatlogs.span_kinds.mapping import infer_span_kind_from_name


def test_python_infers_evaluator_kind():
    assert infer_span_kind_from_name("evaluate_answer_quality") == "EVALUATOR"


def test_evaluator_is_a_recognized_normalized_kind():
    # Previously EVALUATOR was missing from _KINDS, so it normalized to UNKNOWN.
    assert "EVALUATOR" in _KINDS


def test_schema_span_kind_enum_includes_evaluator():
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "neatlogs"
        / "contracts"
        / "v2"
        / "neatlogs-telemetry.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    enum = schema["$defs"]["spanKind"]["enum"]
    assert "EVALUATOR" in enum
