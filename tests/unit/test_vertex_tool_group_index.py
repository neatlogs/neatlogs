"""Regression: vertex_ai must keep every function declaration across multiple
tool groups. It previously indexed tool attributes with `{i + j}` (group index
+ declaration index), which collides across groups and overwrites declarations.
Mirrors the google_genai fix in #114.
"""

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.vertex_ai import _set_input_attributes


def test_multiple_tool_groups_keep_every_function_declaration(
    tracer_provider, in_memory_span_exporter
):
    set_neatlogs_provider(tracer_provider)
    neatlogs.init(
        api_key="test-key",
        instrumentations=[],
        tracer_provider=tracer_provider,
        register_shutdown_handlers=False,
    )

    class Cfg:
        tools = [
            {"function_declarations": [{"name": "a0"}, {"name": "a1"}]},
            {"function_declarations": [{"name": "b0"}, {"name": "b1"}]},
        ]

    span = tracer_provider.get_tracer("t").start_span("test")
    _set_input_attributes(span, contents="hi", kwargs={"config": Cfg()})
    span.end()

    attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
    names = [attrs.get(f"neatlogs.llm.tools.{i}.name") for i in range(4)]
    assert names == ["a0", "a1", "b0", "b1"]
