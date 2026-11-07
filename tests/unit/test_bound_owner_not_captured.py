"""
A decorated method must not export the attributes of the instance or class it is
bound to: ``self`` and ``cls`` are recorded by type only, while the arguments the
caller actually passed are still captured.
"""

import json

from opentelemetry import trace

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.decorators.orchestration import span as neatlogs_span


def _install(tracer_provider):
    trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)


def _input(exporter, name):
    finished = [s for s in exporter.get_finished_spans() if s.name == name][-1]
    return json.loads(finished.attributes["input.value"])


class Agent:
    REGION = "us"
    API_KEY = "sk-proj-class-level"

    def __init__(self):
        self.api_key = "sk-ant-instance-level"

    @neatlogs_span(kind="AGENT", name="agent.answer")
    def answer(self, question):
        return question

    @classmethod
    @neatlogs_span(kind="TOOL", name="agent.from_env")
    def from_env(cls, region):
        return region

    @staticmethod
    @neatlogs_span(kind="TOOL", name="agent.static")
    def static(value):
        return value


def test_method_records_self_by_type_only(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    Agent().answer("hello")

    captured = _input(in_memory_span_exporter, "agent.answer")
    assert captured == {"self": "<Agent>", "question": "hello"}
    assert "sk-ant-instance-level" not in json.dumps(captured)


def test_method_owner_passed_by_keyword_is_recorded_by_type_only(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    Agent.answer(self=Agent(), question="hello")

    captured = _input(in_memory_span_exporter, "agent.answer")
    assert captured == {"self": "<Agent>", "question": "hello"}
    assert "sk-ant-instance-level" not in json.dumps(captured)


def test_classmethod_records_cls_by_type_only(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    Agent.from_env("eu")

    captured = _input(in_memory_span_exporter, "agent.from_env")
    assert captured == {"cls": "<Agent>", "region": "eu"}
    assert "sk-proj-class-level" not in json.dumps(captured)


def test_staticmethod_and_explicit_class_argument(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    Agent.static(Agent)

    captured = _input(in_memory_span_exporter, "agent.static")
    assert captured == {"value": "<class Agent>"}


@neatlogs_span(kind="TOOL", name="plain.self_arg")
def plain_function(self, value):
    return value


def test_plain_function_argument_named_self_is_captured(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    class Config:
        def __init__(self):
            self.region = "eu"

    plain_function(Config(), 1)

    # Not a method, so the caller's own argument is captured like any other.
    assert _input(in_memory_span_exporter, "plain.self_arg") == {
        "self": {"region": "eu"},
        "value": 1,
    }


def test_method_of_a_class_defined_in_a_function(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    class Local:
        def __init__(self):
            self.token = "ghp_local"

        @neatlogs_span(kind="TOOL", name="local.method")
        def run(self):
            return 1

    Local().run()

    assert _input(in_memory_span_exporter, "local.method") == {"self": "<Local>"}


class Utility:
    @staticmethod
    @neatlogs_span(kind="TOOL", name="utility.echo")
    def echo(self):
        return self


def test_staticmethod_argument_named_self_is_captured(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    Utility.echo({"region": "eu"})

    # The caller passed this value explicitly; it is not a bound owner.
    assert _input(in_memory_span_exporter, "utility.echo") == {"self": {"region": "eu"}}


def test_staticmethod_receiving_its_own_class_instance_is_captured(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    class Carrier(Utility):
        def __init__(self):
            self.region = "eu"

    Utility.echo(Carrier())

    # Still an explicit argument to a static method, even though it is a Utility.
    assert _input(in_memory_span_exporter, "utility.echo") == {"self": {"region": "eu"}}


class SubAgent(Agent):
    def __init__(self):
        super().__init__()
        self.extra_token = "ghp_subclass"


def test_inherited_method_on_a_subclass_records_self_by_type_only(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    SubAgent().answer("hi")

    captured = _input(in_memory_span_exporter, "agent.answer")
    assert captured == {"self": "<SubAgent>", "question": "hi"}
    assert "ghp_subclass" not in json.dumps(captured)


def test_plain_object_argument_is_still_captured(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    class Order:
        def __init__(self):
            self.id = 7

    @neatlogs_span(kind="TOOL", name="order.lookup")
    def lookup(order):
        return order.id

    lookup(Order())

    assert _input(in_memory_span_exporter, "order.lookup") == {"order": {"id": 7}}
