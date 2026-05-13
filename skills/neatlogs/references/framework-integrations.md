# Framework Integrations — NeatLogs SDK Reference

Framework-specific integration patterns for the NeatLogs SDK. Covers auto-instrumentation setup, init ordering, install commands, and representative code examples for each supported LLM provider and agent framework.

---

## 1. Integration Approaches

Pick the approach that matches your code. The four approaches below are cumulative — most apps use 2 or 3 of them together.

### 1a. Pure Auto-Instrumentation

For applications that call LLM providers directly. Just add the provider to `instrumentations=[]`. Every SDK call is traced automatically.

### 1b. Auto-Instrumentation + `@span` Decorators

For custom multi-agent orchestration. Add providers to `instrumentations=[]` for LLM call tracing, then use `@span` decorators on your orchestration functions to capture the full call graph.

### 1c. Auto-Instrumentation + `trace()` + `SystemPromptTemplate` / `UserPromptTemplate`

When you want to track prompt templates + variables in the dashboard. Wrap the LLM call in `trace()` and pass `SystemPromptTemplate` / `UserPromptTemplate` instances — the SDK captures the template and compiled variables automatically.

### 1d. Auto-Instrumentation + `bind_templates()`

For CrewAI, where the framework owns the LLM call so `trace()` can't wrap it. Use `bind_templates()` to attach prompt context to the agent's LLM, and `register_crewai_task()` for task-level prompt tracking.

### Complete example combining 1a + 1b + 1c

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="research-app",
    instrumentations=["openai"],
)

from openai import OpenAI  # Import AFTER init()
client = OpenAI()

sys_tpl = SystemPromptTemplate([
    {"role": "system", "content": "You are a {{role}} assistant."}
])
user_tpl = UserPromptTemplate([
    {"role": "user", "content": "{{query}}"}
])


@neatlogs.span(kind="WORKFLOW")
def pipeline(query: str) -> str:
    return researcher(query)


@neatlogs.span(kind="AGENT", name="researcher", role="Research Analyst")
def researcher(query: str) -> str:
    with neatlogs.trace(
        "llm_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        msgs = sys_tpl.compile(role="research") + user_tpl.compile(query=query)
        response = client.chat.completions.create(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content


if __name__ == "__main__":
    result = pipeline("Explain quantum computing")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

This single snippet demonstrates:
- 1a — `instrumentations=["openai"]` auto-traces the LLM call
- 1b — `@span(kind="WORKFLOW")` + `@span(kind="AGENT")` wrap the orchestration
- 1c — `trace(kind="LLM", system_prompt_template=..., user_prompt_template=...)` captures the template + variable bindings

---

## 2. OpenAI

- **Instrumentation key**: `instrumentations=["openai"]`
- **Covers**: The `openai` Python SDK — both `OpenAI()` and `AzureOpenAI()` (part of the same `openai` package)
- **Import order critical**: `neatlogs.init()` BEFORE `from openai import OpenAI`
- **Supports**: sync, async, streaming
- **Install**: `pip install neatlogs[openai]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="my-app",
    instrumentations=["openai"],
)

from openai import OpenAI
client = OpenAI()

sys_tpl = SystemPromptTemplate([
    {"role": "system", "content": "You are a helpful assistant specializing in {{domain}}."}
])
user_tpl = UserPromptTemplate([
    {"role": "user", "content": "Question: {{query}}"}
])


@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    with neatlogs.trace(
        "llm_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        msgs = sys_tpl.compile(domain="science") + user_tpl.compile(query=query)
        response = client.chat.completions.create(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content


if __name__ == "__main__":
    result = run("Explain quantum computing")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

---

## 3. Anthropic

- **Instrumentation key**: `instrumentations=["anthropic"]`
- **Supports**: Extended thinking, streaming, tool use
- **Also works with `AnthropicBedrock`**: still use `instrumentations=["anthropic"]`
- **Install**: `pip install neatlogs[anthropic]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="anthropic-app",
    instrumentations=["anthropic"],
)

from anthropic import Anthropic
client = Anthropic()

sys_tpl = SystemPromptTemplate("You are a market analysis expert for {{industry}}.")
user_tpl = UserPromptTemplate([{"role": "user", "content": "Analyze: {{query}}"}])


@neatlogs.span(kind="AGENT", name="analyst")
def analyst(query: str) -> str:
    with neatlogs.trace(
        "llm_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        sys_str = sys_tpl.compile(industry="technology")
        user_msgs = user_tpl.compile(query=query)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=sys_str,
            messages=user_msgs,
        )
    return response.content[0].text


if __name__ == "__main__":
    result = analyst("Analyze market trends")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

---

## 4. Google GenAI (Gemini)

- **Instrumentation key**: `instrumentations=["google_genai"]`
- **Stricter init ordering**: `neatlogs.init()` must precede `google.genai.Client()` **instantiation**, not just import. The client object caches the transport at construction time.
- **Supports**: sync, streaming, async streaming
- **Install**: `pip install neatlogs[google-genai]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="gemini-app",
    instrumentations=["google_genai"],
)

# Client MUST be created AFTER init()
from google import genai
client = genai.Client(api_key="...")

sys_tpl = SystemPromptTemplate("You are a research assistant for {{domain}}.")
user_tpl = UserPromptTemplate("Research topic: {{topic}}")


@neatlogs.span(kind="AGENT", name="researcher")
def researcher(topic: str) -> str:
    with neatlogs.trace(
        "llm_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        prompt = sys_tpl.compile(domain="science") + " " + user_tpl.compile(topic=topic)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    return response.text


if __name__ == "__main__":
    result = researcher("quantum computing advances")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

See [`troubleshooting.md` §2](troubleshooting.md#2-google-genai-instantiation-ordering) for the wrong-vs-right code on Google GenAI client ordering.

---

## 5. LangChain

- **Instrumentation key**: `instrumentations=["langchain"]`
- **Auto-instruments**: LLM calls, chains, agents, tools, retrievers
- **Works with**: AgentExecutor, ReAct agents, LCEL chains
- **Install**: `pip install neatlogs[langchain]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="langchain-app",
    instrumentations=["langchain"],
)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

sys_tpl = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a helpful research assistant for {{domain}}.",
}])
user_tpl = UserPromptTemplate([{
    "role": "user",
    "content": "Research this topic: {{query}}",
}])


@neatlogs.span(kind="WORKFLOW")
def run_agent(query: str) -> str:
    with neatlogs.trace(
        "research_llm",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        msgs = sys_tpl.compile(domain="science") + user_tpl.compile(query=query)
        response = llm.invoke(msgs)
    return response.content


if __name__ == "__main__":
    result = run_agent("Explain black holes")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

---

## 6. LangGraph

- **Instrumentation key**: `instrumentations=["langchain"]` (LangGraph uses LangChain instrumentation)
- **Tracks**: Graph execution, nodes, tool loops, fan-out/fan-in
- **Install**: `pip install neatlogs[langchain,langgraph]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="langgraph-app",
    instrumentations=["langchain"],
)

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

supervisor_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a supervisor routing research tasks to agents.",
}])
supervisor_user = UserPromptTemplate([{
    "role": "user",
    "content": "Research topic: {{query}}",
}])

researcher_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a research agent for {{domain}}.",
}])
researcher_user = UserPromptTemplate([{
    "role": "user",
    "content": "Research: {{query}}",
}])


def supervisor_node(state):
    with neatlogs.trace(
        "supervisor_llm",
        kind="LLM",
        system_prompt_template=supervisor_sys,
        user_prompt_template=supervisor_user,
    ):
        msgs = supervisor_sys.compile() + supervisor_user.compile(query=state["query"])
        response = llm.invoke(msgs)
    return {"next": response.content}


def researcher_node(state):
    with neatlogs.trace(
        "researcher_llm",
        kind="LLM",
        system_prompt_template=researcher_sys,
        user_prompt_template=researcher_user,
    ):
        msgs = researcher_sys.compile(domain="technology") + researcher_user.compile(query=state["query"])
        response = llm.invoke(msgs)
    return {"result": response.content}


graph = StateGraph(dict)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
# ... add edges ...
app = graph.compile()


@neatlogs.span(kind="WORKFLOW")
def run_pipeline(query: str) -> str:
    result = app.invoke({"query": query})
    return result.get("result", "")


if __name__ == "__main__":
    result = run_pipeline("latest AI trends")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

---

## 7. CrewAI

- **Instrumentation key**: `instrumentations=["crewai", "<provider_key>"]`. Pick the provider key that matches the underlying LLM backing `crewai.LLM(...)`:
  - OpenAI (`OPENAI_API_KEY` + `model="gpt-4o"` / similar) → `"openai"`
  - Azure OpenAI (`model="azure/..."`) → `"azure_ai_inference"`
  - Google GenAI (`model="gemini/..."`) → `"google_genai"`
  - Anthropic (`model="claude-..."`) → `"anthropic"`
- **Use `bind_templates()`** to attach prompt context to agent LLMs
- **Use `register_crewai_task(task, user_tpl, **vars)`** for task-level prompt tracking
- **Install**: `pip install neatlogs[crewai]` (pulls in `crewai >= 1.9.3` and `litellm`)
- **Version note**: SDK pins `crewai >= 1.9.3`. CrewAI API has changed significantly between versions — ensure version compatibility.

> **Why the provider key matters here**: CrewAI dispatches LLM calls internally via LiteLLM. For OpenAI-proper the `openai` instrumentor catches the call; for Azure OpenAI (a different SDK path) you need `azure_ai_inference`, otherwise the LLM call succeeds but no `LLM`-kind span is produced — the trace shows only the Agent parent with no LLM child.

```python
import os
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="crewai-app",
    # Azure OpenAI backend -> azure_ai_inference. Swap for openai / google_genai /
    # anthropic depending on what crewai.LLM(model=...) points at.
    instrumentations=["crewai", "azure_ai_inference"],
)

from crewai import Agent, Task, Crew, LLM


@neatlogs.span(kind="WORKFLOW")
def run_crew() -> str:
    # System template must NOT have required placeholders — bind_templates()
    # calls system_tpl.compile() with no arguments. Pre-render if needed.
    analyst_tpl = SystemPromptTemplate("You are a senior market analyst.")

    # User template for the task
    task_tpl = UserPromptTemplate("Analyze {{topic}} trends for {{year}}.")

    # Create and bind LLM — returns a new LLM instance with template context attached.
    # Using Azure OpenAI here; for OpenAI-proper use model="gpt-4o" without api_version/base_url.
    llm = LLM(
        model="azure/gpt-5-nano",
        api_key=os.environ["AZURE_API_KEY"],
        base_url=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    bound_llm = neatlogs.bind_templates(llm, analyst_tpl)

    analyst = Agent(
        role="Market Analyst",
        goal="Provide market analysis",
        backstory="Expert analyst with 10 years experience",
        llm=bound_llm,
    )

    task = Task(
        description="Analyze AI chip market trends",
        expected_output="Market analysis report",
        agent=analyst,
    )
    neatlogs.register_crewai_task(task, task_tpl, topic="AI chips", year="2025")

    crew = Crew(agents=[analyst], tasks=[task])
    return str(crew.kickoff())


if __name__ == "__main__":
    result = run_crew()
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

---

## 8. LiteLLM

- **Instrumentation key**: `instrumentations=["litellm"]`
- **Auto-instruments**: LiteLLM's unified `completion()` API across all providers it supports
- **Install**: `pip install neatlogs[litellm]`

```python
import os
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

os.environ.setdefault("GEMINI_API_KEY", "...")  # or whichever provider you target

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="litellm-app",
    instrumentations=["litellm"],
)

# Import AFTER init() so the instrumentor patches the litellm module.
from litellm import completion

sys_tpl = SystemPromptTemplate([
    {"role": "system", "content": "You are a helpful assistant."}
])
user_tpl = UserPromptTemplate([
    {"role": "user", "content": "{{query}}"}
])


@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    with neatlogs.trace(
        "llm_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        msgs = sys_tpl.compile() + user_tpl.compile(query=query)
        response = completion(model="gemini/gemini-2.5-flash", messages=msgs)
    return response.choices[0].message.content


if __name__ == "__main__":
    result = run("Write one short, family-friendly programming joke.")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```

> **Provider routing**: the `model=` argument uses LiteLLM's `<provider>/<model>` convention (`openai/gpt-4o`, `gemini/gemini-2.5-flash`, `anthropic/claude-sonnet-4-6`, etc.). The matching provider API key must be in the environment.

---

## 9. Multi-Provider

When an application fans out across multiple LLM providers, list all of them in `instrumentations=[]`. Each provider's LLM calls are auto-instrumented independently; wrap each call in a separate `trace()` block so prompt templates get attached to the right span.

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="multi-provider-app",
    instrumentations=["openai", "anthropic", "google_genai"],
)

from openai import OpenAI
from anthropic import Anthropic
from google import genai

openai_client = OpenAI()
anthropic_client = Anthropic()
gemini_client = genai.Client(api_key="...")

sys_tpl = SystemPromptTemplate([
    {"role": "system", "content": "You are an expert analyst."}
])
user_tpl = UserPromptTemplate([
    {"role": "user", "content": "{{query}}"}
])


@neatlogs.span(kind="WORKFLOW")
def multi_model_pipeline(query: str) -> dict:
    # OpenAI chat API expects a list of {role, content} dicts.
    with neatlogs.trace(
        "openai_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        msgs = sys_tpl.compile() + user_tpl.compile(query=query)
        gpt_response = openai_client.chat.completions.create(model="gpt-4o", messages=msgs)

    # Anthropic expects system as a string and messages as a list.
    with neatlogs.trace(
        "claude_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        sys_str = sys_tpl.compile()[0]["content"]
        user_msgs = user_tpl.compile(query=query)
        claude_response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=sys_str,
            messages=user_msgs,
        )

    # Gemini generate_content accepts a single concatenated string.
    with neatlogs.trace(
        "gemini_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        sys_str = sys_tpl.compile()[0]["content"]
        user_str = user_tpl.compile(query=query)[0]["content"]
        gemini_response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{sys_str}\n\n{user_str}",
        )

    return {
        "gpt": gpt_response.choices[0].message.content,
        "claude": claude_response.content[0].text,
        "gemini": gemini_response.text,
    }


if __name__ == "__main__":
    result = multi_model_pipeline("Compare approaches to AGI")
    print(result)
    neatlogs.flush()
    neatlogs.shutdown()
```
