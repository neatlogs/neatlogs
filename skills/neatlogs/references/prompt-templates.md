# Prompt Templates — NeatLogs SDK Reference

Prompt template tracking and management in the NeatLogs SDK. Covers local template classes (`SystemPromptTemplate`, `UserPromptTemplate`), integration with `trace()` and `bind_templates()`, CrewAI task-level tracking, and the server-side Prompt Management API.

---

## 1. `SystemPromptTemplate` Class

System / AI instruction prompt with `{{variable}}` placeholders.

- Constructor accepts a **string** OR a **list of message dicts**
- `.compile(**variables)` renders the template, **sets prompt context in OTel for automatic span capture**, and returns:
  - A `str` if constructed with a string template
  - A `List[Dict[str, str]]` (message list) if constructed with a message list — ready to pass directly to `messages=` in OpenAI / Anthropic calls
- `.variables` property lists extracted variable names

```python
from neatlogs import SystemPromptTemplate

# String form
sys_tpl = SystemPromptTemplate("You are a {{role}} assistant specialized in {{domain}}.")

# Message list form (preferred for chat models)
sys_tpl = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a {{role}} assistant specialized in {{domain}}."
}])

# Check variables
print(sys_tpl.variables)  # ["role", "domain"]

# Compile (renders template + sets OTel context)
messages = sys_tpl.compile(role="research", domain="quantum physics")
```

> `PromptTemplate` is kept as a backward-compatible alias for `SystemPromptTemplate`. New code should use `SystemPromptTemplate`.

---

## 2. `UserPromptTemplate` Class

Same API as `SystemPromptTemplate` but for the user / human turn.

```python
from neatlogs import UserPromptTemplate

user_tpl = UserPromptTemplate([{
    "role": "user",
    "content": "Research topic: {{topic}}\nFocus areas: {{focus}}"
}])

user_msgs = user_tpl.compile(topic="quantum entanglement", focus="recent experiments")
```

---

## 3. Using Templates with `trace()`

Pass `system_prompt_template=` and `user_prompt_template=` to `trace()` for automatic capture on LLM spans. **IMPORTANT**: Call `.compile()` **INSIDE** the `trace()` context for variable bindings to be captured.

> **`neatlogs.SystemPromptTemplate` vs framework prompt templates**: `neatlogs.SystemPromptTemplate` is NeatLogs' own class for template *tracking* in the dashboard — it is independent of LangChain's `ChatPromptTemplate`, OpenAI prompt strings, etc. Use `neatlogs.SystemPromptTemplate` alongside whatever framework prompt class your code already uses.

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="research",
    instrumentations=["openai"],
)

from openai import OpenAI  # Import AFTER init() for auto-instrumentation

client = OpenAI()

sys_tpl = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a {{role}} assistant. Always be thorough."
}])

user_tpl = UserPromptTemplate([{
    "role": "user",
    "content": "Research: {{query}}"
}])

@neatlogs.span(kind="AGENT", name="researcher")
def researcher_agent(query: str) -> str:
    with neatlogs.trace("research_llm", kind="LLM",
                        system_prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile(role="research") + user_tpl.compile(query=query)
        response = client.chat.completions.create(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content
```

> The legacy kwargs `prompt_template=` / `prompt_variables=` still work as aliases for `system_prompt_template=` / `system_prompt_variables=`.

### Anti-Pattern

```python
# WRONG — compile() outside trace() context, variable bindings not captured
msgs = sys_tpl.compile(role="research")
with neatlogs.trace("llm_call", kind="LLM", system_prompt_template=sys_tpl):
    response = client.chat.completions.create(model="gpt-4o", messages=msgs)

# RIGHT — compile() inside trace() context
with neatlogs.trace("llm_call", kind="LLM", system_prompt_template=sys_tpl):
    msgs = sys_tpl.compile(role="research")
    response = client.chat.completions.create(model="gpt-4o", messages=msgs)
```

---

## 4. `bind_templates()` — for CrewAI

When a framework (CrewAI) owns the LLM calls, you can't wrap them in `trace()`. Use `bind_templates()` to attach prompt context that gets injected automatically before every LLM call.

```python
neatlogs.bind_templates(llm, system_tpl, user_tpl=None, **variables)
```

> Outside CrewAI (plain LangChain or direct SDK usage), prefer `trace(system_prompt_template=...)` — it's simpler and gives you explicit control over span boundaries.

### CrewAI Example

```python
from crewai import Agent, Task, Crew, LLM
import neatlogs
from neatlogs import SystemPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="marketing",
    instrumentations=["crewai", "openai"],
)

# System template must NOT have required placeholders — bind_templates()
# calls system_tpl.compile() with no arguments. Pre-render if needed.
analyst_tpl = SystemPromptTemplate("You are a senior market analyst for the tech industry.")
llm = LLM(model="gpt-4o", api_key="...")

# Bind template to LLM — returns a new LLM instance
bound_llm = neatlogs.bind_templates(llm, analyst_tpl)

analyst = Agent(
    role="Market Analyst",
    goal="Analyze market trends",
    llm=bound_llm,  # Use the bound LLM
    # ...
)
```

---

## 5. `register_crewai_task()` for Task-Level Prompt Tracking

Attaches a `UserPromptTemplate` to a CrewAI `Task` so the template is stamped on the `CREWAI_TASK` span when the task executes.

```python
neatlogs.register_crewai_task(task, user_tpl, **variables)
```

When the task executes, the user template (and any variables you pass) is shown in the dashboard as the **Task Prompt** on the corresponding Agent span, alongside the **Agent Prompt** that came from `bind_templates()`.

```python
from crewai import Task
from neatlogs import UserPromptTemplate

user_tpl = UserPromptTemplate("Analyze the market for {{product}} in {{region}}.")

task = Task(
    description="Analyze the market...",
    expected_output="A market analysis report",
    agent=analyst,
)

neatlogs.register_crewai_task(task, user_tpl, product="AI chips", region="North America")
```

---

## 6. Prompt Management API (Server-Side)

The Prompt Management API stores and retrieves prompt templates from the NeatLogs backend. Requires a valid `NEATLOGS_API_KEY`.

### Retrieving Prompts — Sync

```python
import neatlogs

# Get a prompt by label (returns PromptHandle). `label` OR `version` — not both.
prompt = neatlogs.get_prompt(name="research-agent", label="production")

# By version
prompt = neatlogs.get_prompt(name="research-agent", version=3)

# For chat-style prompts (with message list)
chat_prompt = neatlogs.get_prompt(name="chatbot", label="production", type="chat")

# Use a shorter cache TTL for hot-reload scenarios
fresh = neatlogs.get_prompt(name="research-agent", label="production", cache_ttl_seconds=5)

# Access properties
print(prompt.name)        # "research-agent"
print(prompt.version)     # 3
print(prompt.content)     # Raw template string
print(prompt.messages)    # Message list (if chat-type)
print(prompt.labels)      # ["production"]
print(prompt.config)      # Config dict
print(prompt.type)        # "text" or "chat"
print(prompt.updated_at)  # ISO timestamp

# Compile with variables (pass a dict, not kwargs)
compiled_str = prompt.compile(variables={"topic": "quantum computing"})
compiled_messages = prompt.compile_messages(variables={"topic": "quantum computing"})
```

```python
# fetch_prompt() always returns the latest version from the backend
latest = neatlogs.fetch_prompt(name="research-agent", label="production")
```

### Retrieving Prompts — Async

```python
import asyncio
import neatlogs

async def main():
    prompt = await neatlogs.aget_prompt(
        name="research-agent",
        label="production",
    )
    compiled = prompt.compile(variables={"topic": "quantum computing"})
    ...

asyncio.run(main())
```

`aget_prompt()` is the async sibling of `get_prompt()` — same arguments, same return type, no thread-pool hop needed in async apps.

### Creating and Managing Prompts

```python
# Create — `prompt=` is the template text OR a list of message dicts,
# `labels=` is required.
new_prompt = neatlogs.create_prompt(
    name="research-agent",
    prompt="You are a research assistant for {{topic}}.",
    type="text",                  # or "chat" with a messages list
    labels=["staging"],
    tags=["v1"],                  # optional
    config={"temperature": 0.2},  # optional
    commit_message="initial version",  # optional
)

# Move labels to a specific version (does NOT update content)
neatlogs.update_prompt(name="research-agent", version=1, new_labels=["production"])

# Save new content as a new version — pass EITHER `content=` (text) OR
# `messages=` (chat message list)
neatlogs.save_as_version(
    prompt_name="research-agent",
    content="Updated research assistant for {{topic}}.",
    labels=["staging"],
)

# Chat-style
neatlogs.save_as_version(
    prompt_name="chatbot",
    messages=[{"role": "system", "content": "You help with {{topic}}."}],
    labels=["staging"],
)

# List all prompts — supports filters
all_prompts = neatlogs.list_prompts(name=None, label=None, limit=100, offset=0)

# Delete a specific version
neatlogs.delete_prompt(name="research-agent", version=1)

# Remove a label/tag from a version
neatlogs.remove_tag(name="research-agent", version=2, tag="staging")
```

### Error Handling

```python
from neatlogs import PromptNotFoundError, PromptApiError, PromptClientError

try:
    prompt = neatlogs.get_prompt(name="nonexistent")
except PromptNotFoundError:
    print("Prompt not found")
except PromptApiError as e:
    # Backend returned an error response
    print(f"API error: {e}")
except PromptClientError as e:
    # Other client-side errors (transport, parsing, etc.)
    print(f"Client error: {e}")
```

For advanced use cases, instantiate `PromptClient` or `AsyncPromptClient` directly — both are exported from `neatlogs`.

### When to Use

- Managing prompts centrally across environments (dev, staging, production)
- A/B testing prompt versions via labels
- Sharing prompts between team members
- Version-controlling prompts server-side

> Requires a NeatLogs backend connection with a valid API key. Without one, these functions raise `PromptApiError`.
