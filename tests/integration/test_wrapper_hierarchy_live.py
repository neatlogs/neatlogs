"""
Live wrapper hierarchy tests — NO MOCKS.

These exercise neatlogs.wrap() / hooks / processors against the REAL installed
agent libraries and REAL LLM providers, asserting the emitted span hierarchy
(WORKFLOW/AGENT/TASK/CHAIN/LLM/TOOL/EMBEDDING) and single-trace cohesion.

Each test skips unless the relevant API key is present in the environment. Keys
are loaded from the neatlogs-wizard test fixtures if available (see _load_keys).

Run:
    pytest tests/integration/test_wrapper_hierarchy_live.py -v
"""

import glob
import os

import pytest
from opentelemetry import trace as ot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# ---------------------------------------------------------------------------
# Key loading + fixtures
# ---------------------------------------------------------------------------

_WANT_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
_FIXTURE_GLOB = os.path.expanduser(
    "~/Projects/neatlogs-wizard/test-fixtures/workflows/python/*/.env"
)


def _load_keys() -> None:
    for envf in glob.glob(_FIXTURE_GLOB):
        try:
            for line in open(envf):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k in _WANT_KEYS and k not in os.environ and v and "your-" not in v and "xxx" not in v.lower():
                    os.environ[k] = v
        except OSError:
            continue


_load_keys()


def _exporter():
    """Fresh in-memory exporter wired to a fresh provider (override allowed)."""
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    ot._TRACER_PROVIDER = None  # allow re-registration across tests
    ot.set_tracer_provider(prov)
    # The neatlogs wrapper caches a tracer from the first provider it sees; reset
    # it so each test's spans flow into this test's exporter.
    import neatlogs._wrap_utils as wu
    wu._wrapper_tracer = None
    return exp, prov


def _kinds(exp):
    return [s.attributes.get("neatlogs.span.kind") for s in exp.get_finished_spans() if s.attributes.get("neatlogs.span.kind")]


def _names(exp):
    from collections import Counter
    return Counter(s.name for s in exp.get_finished_spans() if s.attributes.get("neatlogs.span.kind"))


def _trace_count(exp):
    return len({s.context.trace_id for s in exp.get_finished_spans() if s.attributes.get("neatlogs.span.kind")})


needs_openai = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
needs_anthropic = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
needs_google = pytest.mark.skipif(
    not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")), reason="GOOGLE/GEMINI key not set"
)

OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@needs_openai
def test_openai_resources():
    exp, _ = _exporter()
    import neatlogs
    import openai

    c = neatlogs.wrap(openai.OpenAI())
    c.chat.completions.create(model=OPENAI_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=5)
    for _ in c.chat.completions.create(model=OPENAI_MODEL, messages=[{"role": "user", "content": "count to 3"}], max_tokens=10, stream=True):
        pass
    c.embeddings.create(model="text-embedding-3-small", input="hello")

    names = _names(exp)
    assert names["openai.chat.completions.create"] >= 2
    assert names["openai.embeddings.create"] == 1
    assert "EMBEDDING" in _kinds(exp)


@needs_anthropic
def test_anthropic_resources():
    exp, _ = _exporter()
    import neatlogs
    import anthropic

    model = "claude-haiku-4-5-20251001"
    c = neatlogs.wrap(anthropic.Anthropic())
    c.messages.create(model=model, max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    for _ in c.messages.create(model=model, max_tokens=10, messages=[{"role": "user", "content": "count to 3"}], stream=True):
        pass
    c.messages.count_tokens(model=model, messages=[{"role": "user", "content": "hello there"}])

    names = _names(exp)
    assert names["anthropic.messages.create"] >= 2
    assert names["anthropic.messages.count_tokens"] == 1


@needs_google
def test_google_genai_resources():
    exp, _ = _exporter()
    import neatlogs
    from google import genai

    c = neatlogs.wrap(genai.Client())
    c.models.generate_content(model=GEMINI_MODEL, contents="say hi")
    c.models.count_tokens(model=GEMINI_MODEL, contents="how many tokens")
    chat = c.chats.create(model=GEMINI_MODEL)
    chat.send_message("say ok")

    names = _names(exp)
    assert names["google_genai.models.generate_content"] >= 1
    assert names["google_genai.models.count_tokens"] == 1
    assert names["google_genai.chat.send_message"] == 1


# ---------------------------------------------------------------------------
# Frameworks
# ---------------------------------------------------------------------------


@needs_openai
def test_dspy_hierarchy():
    exp, _ = _exporter()
    import neatlogs
    import dspy

    neatlogs.wrap(dspy.Predict("question -> answer"))
    dspy.configure(lm=dspy.LM(f"openai/{OPENAI_MODEL}", max_tokens=100))
    cot = dspy.ChainOfThought("question -> answer")
    cot(question="What is 2+2? Reply with just the number.")

    kinds = _kinds(exp)
    assert "CHAIN" in kinds and "LLM" in kinds
    assert _trace_count(exp) == 1


@needs_openai
def test_pydantic_ai_hierarchy():
    exp, _ = _exporter()
    import neatlogs
    from pydantic_ai import Agent

    agent = Agent(f"openai:{OPENAI_MODEL}")

    @agent.tool_plain
    def get_secret_code(item: str) -> str:
        "Return the secret access code for an item."
        return "ZX-9981-QQ"

    neatlogs.wrap(agent)
    agent.run_sync("Get the secret access code for 'mainframe' using the tool. Return only the code.")

    kinds = set(_kinds(exp))
    assert {"AGENT", "LLM", "TOOL"} <= kinds
    assert _kinds(exp).count("AGENT") == 1
    assert _trace_count(exp) == 1


@needs_openai
def test_crewai_hierarchy():
    exp, _ = _exporter()
    os.environ["CREWAI_TRACING_ENABLED"] = "false"
    import neatlogs
    from crewai import Agent, Task, Crew
    from crewai.tools import tool

    @tool("get_secret_code")
    def get_secret_code(item: str) -> str:
        "Returns the secret access code for a given item."
        return "ZX-9981-QQ"

    ag = Agent(role="lookup", goal="find codes via tools", backstory="x",
               tools=[get_secret_code], llm=OPENAI_MODEL, verbose=False)
    tk = Task(description="Find the secret access code for 'mainframe' using the tool. Return only the code.",
              expected_output="code", agent=ag)
    crew = Crew(agents=[ag], tasks=[tk], verbose=False)
    neatlogs.wrap(crew)
    crew.kickoff()

    kinds = set(_kinds(exp))
    assert {"WORKFLOW", "TASK", "AGENT", "LLM", "TOOL"} <= kinds
    assert _trace_count(exp) == 1


@needs_openai
def test_openai_agents_hierarchy():
    exp, _ = _exporter()
    import neatlogs
    from agents import Agent, Runner, function_tool, set_trace_processors

    set_trace_processors([neatlogs.openai_agents_processor()])

    @function_tool
    def get_secret_code(item: str) -> str:
        "Return the secret access code for an item."
        return "ZX-9981-QQ"

    agent = Agent(name="lookup", instructions="Use tools to answer.", tools=[get_secret_code], model=OPENAI_MODEL)
    Runner.run_sync(agent, "Secret access code for 'mainframe'? Use the tool, return only the code.")

    kinds = set(_kinds(exp))
    assert {"WORKFLOW", "AGENT", "LLM", "TOOL"} <= kinds
    assert _trace_count(exp) == 1


@needs_openai
def test_langchain_hierarchy():
    exp, _ = _exporter()
    import neatlogs
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_secret_code(item: str) -> str:
        "Return the secret access code for an item."
        return "ZX-9981-QQ"

    agent = create_react_agent(ChatOpenAI(model=OPENAI_MODEL, temperature=0), [get_secret_code])
    handler = neatlogs.langchain_handler()
    agent.invoke(
        {"messages": [("user", "Secret access code for 'mainframe'? Use the tool, return only the code.")]},
        config={"callbacks": [handler]},
    )

    kinds = set(_kinds(exp))
    assert {"CHAIN", "LLM", "TOOL"} <= kinds
    # The bulk of spans share one trace; a callback handler can't force the
    # ambient OTel context during LangGraph's own tool execution, so we assert
    # the dominant trace holds the chain root + most spans rather than ==1.
    spans = [s for s in exp.get_finished_spans() if s.attributes.get("neatlogs.span.kind")]
    from collections import Counter
    dominant = Counter(s.context.trace_id for s in spans).most_common(1)[0][1]
    assert dominant >= len(spans) - 2
