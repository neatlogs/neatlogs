# Framework Integrations — NeatLogs SDK v3 Reference

Framework-specific integration patterns for NeatLogs SDK v3. Covers auto-instrumentation setup, init ordering, install commands, and representative code examples for each supported LLM provider and agent framework.

---

## 1. Integration Approaches (Decision Tree)

Choose the approach that matches your application architecture:

### 1a. Pure Auto-Instrumentation

For applications that call LLM providers directly (OpenAI, Anthropic, Google GenAI, etc.). Just add the provider to `instrumentations=[]`. No decorators needed for LLM calls — every SDK call is traced automatically.

```python
neatlogs.init(instrumentations=["openai"])
```

### 1b. Auto-Instrumentation + `@span` Decorators

For custom multi-agent orchestration. Add providers to `instrumentations=[]` for LLM call tracing, then use `@span` decorators on your orchestration functions to capture the full call graph.

```python
neatlogs.init(instrumentations=["openai", "anthropic"])

@neatlogs.span(kind="WORKFLOW")
def pipeline(query: str) -> str:
    result_a = agent_a(query)       # @span(kind="AGENT")
    result_b = agent_b(result_a)    # @span(kind="AGENT")
    return result_b
```

### 1c. Auto-Instrumentation + `bind_templates`

For framework-managed LLM calls (e.g. CrewAI). Add `"crewai"` to instrumentations, use `bind_templates()` to attach prompt context to framework-owned LLMs, and `register_crewai_task()` for task-level prompt tracking.

```python
neatlogs.init(instrumentations=["openai", "crewai", "langchain"])
bound_llm = neatlogs.bind_templates(llm, system_tpl)
neatlogs.register_crewai_task(task, user_tpl, topic="AI chips")
```

### 1d. Auto-Instrumentation + `trace()` + `SystemPromptTemplate`

For any integration where you want to track prompt templates and variables in the dashboard. Wrap your LLM call in `trace()` and pass `SystemPromptTemplate` instances — the SDK captures the template and compiled variables automatically.

```python
neatlogs.init(instrumentations=["openai"])

sys_tpl = SystemPromptTemplate("You are a {{role}} assistant.")
user_tpl = UserPromptTemplate("{{query}}")

with neatlogs.trace("llm_call", kind="LLM", prompt_template=sys_tpl, user_prompt_template=user_tpl):
    msgs = sys_tpl.compile(role="research") + user_tpl.compile(query=query)
    response = client.chat.completions.create(model="gpt-4o", messages=msgs)
```

---

## 2. OpenAI

- **Instrumentation key**: `instrumentations=["openai"]`
- **Covers**: The `openai` Python SDK — both `OpenAI()` and `AzureOpenAI()` (which is part of the same `openai` package)
- **Import order critical**: `neatlogs.init()` BEFORE `from openai import OpenAI`
- **Supports**: Sync, async, streaming
- **Install**: `pip install neatlogs[openai]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # or set NEATLOGS_API_KEY env var
    workflow_name="my-app",
    instrumentations=["openai"],
)

from openai import OpenAI

client = OpenAI()

sys_tpl = SystemPromptTemplate("You are a helpful assistant specializing in {{domain}}.")
user_tpl = UserPromptTemplate("Question: {{query}}")

@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    with neatlogs.trace("llm_call", kind="LLM",
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile(domain="science") + user_tpl.compile(query=query)
        response = client.chat.completions.create(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content

result = run("Explain quantum computing")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 3. Anthropic

- **Instrumentation key**: `instrumentations=["anthropic"]`
- **Supports**: Extended thinking, streaming, tool use
- **Also works with `AnthropicBedrock`**: Still use `instrumentations=["anthropic"]`
- **Install**: `pip install neatlogs[anthropic]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # or set NEATLOGS_API_KEY env var
    workflow_name="anthropic-app",
    instrumentations=["anthropic"],
)

from anthropic import Anthropic

client = Anthropic()

sys_tpl = SystemPromptTemplate("You are a market analysis expert for {{industry}}.")
user_tpl = UserPromptTemplate([{"role": "user", "content": "Analyze: {{query}}"}])

@neatlogs.span(kind="AGENT", name="analyst")
def analyst(query: str) -> str:
    with neatlogs.trace("llm_call", kind="LLM",
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        sys_str = sys_tpl.compile(industry="technology")
        user_msgs = user_tpl.compile(query=query)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=sys_str,
            messages=user_msgs,
        )
    return response.content[0].text

result = analyst("Analyze market trends")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 4. Google GenAI (Gemini)

- **Instrumentation key**: `instrumentations=["google_genai"]`
- **Stricter init ordering**: `neatlogs.init()` must precede `google.genai.Client()` **instantiation**, not just import. The client object caches the transport at construction time.
- **Supports**: Sync, streaming, async streaming
- **Install**: `pip install neatlogs[google-genai]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # or set NEATLOGS_API_KEY env var
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
    with neatlogs.trace("llm_call", kind="LLM",
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        prompt = sys_tpl.compile(domain="science") + " " + user_tpl.compile(topic=topic)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    return response.text

result = researcher("quantum computing advances")
neatlogs.flush()
neatlogs.shutdown()
```

### Wrong vs Right (Google GenAI)

See [`troubleshooting.md` §2](troubleshooting.md#2-google-genai-instantiation-ordering) for the full wrong/right code examples on Google GenAI client ordering.

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
    api_key="...",  # or set NEATLOGS_API_KEY env var
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
    with neatlogs.trace("research_llm", kind="LLM",
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile(domain="science") + user_tpl.compile(query=query)
        response = llm.invoke(msgs)
    return response.content

result = run_agent("Explain black holes")
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
    api_key="...",  # or set NEATLOGS_API_KEY env var
    workflow_name="langgraph-app",
    instrumentations=["langchain"],
)

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Define templates per node — system template + user template with variables
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
    with neatlogs.trace("supervisor_llm", kind="LLM",
                        prompt_template=supervisor_sys,
                        user_prompt_template=supervisor_user):
        msgs = supervisor_sys.compile() + supervisor_user.compile(query=state["query"])
        response = llm.invoke(msgs)
    return {"next": response.content}

def researcher_node(state):
    with neatlogs.trace("researcher_llm", kind="LLM",
                        prompt_template=researcher_sys,
                        user_prompt_template=researcher_user):
        msgs = researcher_sys.compile(domain="technology") + researcher_user.compile(query=state["query"])
        response = llm.invoke(msgs)
    return {"result": response.content}

# Build graph
graph = StateGraph(dict)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
# ... add edges ...
app = graph.compile()

@neatlogs.span(kind="WORKFLOW")
def run_pipeline(query: str) -> str:
    result = app.invoke({"query": query})
    return result.get("result", "")

result = run_pipeline("latest AI trends")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 7. CrewAI

- **Instrumentation key**: start with `instrumentations=["crewai"]`. CrewAI auto-loads LiteLLM. If the CrewAI LLM is backed by a direct provider SDK, also add that provider key: Azure OpenAI / Azure AI Inference → `"azure_ai_inference"`, OpenAI → `"openai"`, Google GenAI → `"google_genai"`, Anthropic → `"anthropic"`.
- **Use `bind_templates()`** to attach prompt context to agent LLMs
- **Use `register_crewai_task(task, user_tpl, **vars)`** for task-level prompt tracking
- **Install**: `pip install neatlogs[crewai]` (pulls in `crewai >= 1.9.3` and `litellm`)
- **Note**: SDK pins `crewai >= 1.9.3`. CrewAI API has changed significantly between versions — ensure version compatibility.

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # or set NEATLOGS_API_KEY env var
    workflow_name="crewai-app",
    instrumentations=["crewai", "openai"],  # Azure OpenAI / Azure AI Inference: use "azure_ai_inference" instead of "openai"
)

from crewai import Agent, Task, Crew, LLM

# System template for the agent's LLM — must NOT have required placeholders
# because bind_templates() calls system_tpl.compile() with no arguments.
# Pre-render the template string if you need dynamic values.
analyst_tpl = SystemPromptTemplate("You are a senior market analyst.")

# User template for the task
task_tpl = UserPromptTemplate("Analyze {{topic}} trends for {{year}}.")

# Create and bind LLM
llm = LLM(model="gpt-4o")
bound_llm = neatlogs.bind_templates(llm, analyst_tpl)

# Create agent with bound LLM
analyst = Agent(
    role="Market Analyst",
    goal="Provide market analysis",
    backstory="Expert analyst with 10 years experience",
    llm=bound_llm,
)

# Create task and register template
task = Task(
    description="Analyze AI chip market trends",
    expected_output="Market analysis report",
    agent=analyst,
)
neatlogs.register_crewai_task(task, task_tpl, topic="AI chips", year="2025")

# Run crew
crew = Crew(agents=[analyst], tasks=[task])

@neatlogs.span(kind="WORKFLOW")
def run_crew():
    return crew.kickoff()

result = run_crew()

neatlogs.flush()
neatlogs.shutdown()
```

---

## 8. LiteLLM

- **Instrumentation key**: `instrumentations=["litellm"]`
- **Auto-instruments**: LiteLLM's unified API across all providers
- **Install**: `pip install neatlogs[litellm]`

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # or set NEATLOGS_API_KEY env var
    workflow_name="litellm-app",
    instrumentations=["litellm"],
)

from litellm import completion

sys_tpl = SystemPromptTemplate("You are a helpful assistant.")
user_tpl = UserPromptTemplate("{{query}}")

@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    with neatlogs.trace("llm_call", kind="LLM",
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile() + user_tpl.compile(query=query)
        response = completion(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content

result = run("Hello!")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 9. Multi-Provider

When using multiple LLM providers, list all of them in `instrumentations`:

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # or set NEATLOGS_API_KEY env var
    workflow_name="multi-provider-app",
    instrumentations=["openai", "anthropic", "google_genai", "langchain"],
)

from openai import OpenAI
from anthropic import Anthropic
from google import genai

openai_client = OpenAI()
anthropic_client = Anthropic()
gemini_client = genai.Client(api_key="...")

sys_tpl = SystemPromptTemplate("You are an expert analyst.")
user_tpl = UserPromptTemplate("{{query}}")

@neatlogs.span(kind="WORKFLOW")
def multi_model_pipeline(query: str) -> dict:
    # Each provider's LLM calls are auto-instrumented independently
    with neatlogs.trace("openai_call", kind="LLM", prompt_template=sys_tpl, user_prompt_template=user_tpl):
        msgs = sys_tpl.compile() + user_tpl.compile(query=query)
        gpt_response = openai_client.chat.completions.create(model="gpt-4o", messages=msgs)

    claude_response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    gemini_response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
    )
    return {
        "gpt": gpt_response.choices[0].message.content,
        "claude": claude_response.content[0].text,
        "gemini": gemini_response.text,
    }

result = multi_model_pipeline("Compare approaches to AGI")
neatlogs.flush()
neatlogs.shutdown()
```
