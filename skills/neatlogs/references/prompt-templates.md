# Prompt Templates — NeatLogs SDK v3 Reference

Prompt template tracking and management in NeatLogs SDK v3. Covers local template classes (`SystemPromptTemplate`, `UserPromptTemplate`), integration with `trace()` and `bind_templates()`, CrewAI task-level tracking, and the server-side Prompt Management API.

---

## 1. `SystemPromptTemplate` Class

System/AI instruction prompt with `{{variable}}` placeholders.

- Constructor accepts a **string** OR a **list of message dicts**
- `.compile(**variables)` renders the template, **sets prompt context in OTel for automatic span capture**, and returns:
  - A `str` if constructed with a string template
  - A `List[Dict[str, str]]` (message list) if constructed with a message list — ready to pass directly to `messages=` in OpenAI/Anthropic calls
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

---

## 2. `UserPromptTemplate` Class

Same API as `SystemPromptTemplate` but for the user/human turn.

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

Pass `prompt_template=` and `user_prompt_template=` to `trace()` for automatic capture on LLM spans. **IMPORTANT**: Call `.compile()` **INSIDE** the `trace()` context for variable bindings to be captured.

> **`neatlogs.SystemPromptTemplate` vs framework prompt templates**: `neatlogs.SystemPromptTemplate` is NeatLogs' own class for template *tracking* in the dashboard — it is independent of LangChain's `ChatPromptTemplate`, OpenAI prompt strings, etc. Use `neatlogs.SystemPromptTemplate` alongside whatever framework prompt class your code already uses.

Pattern from sdk-v3 examples:

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(api_key="...", workflow_name="research", instrumentations=["openai"])

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
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile(role="research") + user_tpl.compile(query=query)
        response = client.chat.completions.create(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content
```

### Anti-Pattern

```python
# WRONG — compile() outside trace() context, variable bindings not captured
msgs = sys_tpl.compile(role="research")
with neatlogs.trace("llm_call", kind="LLM", prompt_template=sys_tpl):
    response = client.chat.completions.create(model="gpt-4o", messages=msgs)

# RIGHT — compile() inside trace() context
with neatlogs.trace("llm_call", kind="LLM", prompt_template=sys_tpl):
    msgs = sys_tpl.compile(role="research")
    response = client.chat.completions.create(model="gpt-4o", messages=msgs)
```

---

## 4. `bind_templates()` for Framework-Managed LLMs

When a framework (like CrewAI) owns the LLM calls, you can't wrap them in `trace()`. Use `bind_templates()` to attach prompt context that gets injected automatically before every LLM call.

```python
neatlogs.bind_templates(llm, system_tpl, user_tpl=None, **variables)
```

- Returns a **new LLM instance** (shallow copy) with template context attached
- SDK v3 supports both `.invoke()` (LangChain `BaseChatModel`) and `.call()` (CrewAI `crewai.LLM`) — the binder checks for `.invoke()` first, then falls back to `.call()`
- When the framework calls the bound LLM, the binder:
  1. Sets `neatlogs.prompt_template` and `neatlogs.user_prompt_template` in OTel context
  2. Calls the instrumented LLM method (creates the LLM span)
  3. Span processor reads the template context and attaches it to the span

### CrewAI Example

```python
from crewai import Agent, Task, Crew, LLM
import neatlogs
from neatlogs import SystemPromptTemplate

neatlogs.init(
    api_key="...",
    workflow_name="marketing",
    instrumentations=["openai", "crewai", "langchain"],
)

# System template must NOT have required placeholders — bind_templates()
# calls system_tpl.compile() with no arguments. Pre-render if needed.
analyst_tpl = SystemPromptTemplate("You are a senior market analyst for the tech industry.")
llm = LLM(model="azure/gpt-4o", api_key="...", base_url="...")

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

- Stores `task.id → (template_str, vars_json)` in a thread-safe registry
- Span processor reads and clears the entry when the task's `_execute_core` AGENT span ends

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

### Retrieving Prompts

```python
import neatlogs

# Get a prompt by name (returns PromptHandle)
prompt = neatlogs.get_prompt(name="research-agent", label="production")

# Access properties
print(prompt.name)        # "research-agent"
print(prompt.version)     # 3
print(prompt.content)     # Raw template string
print(prompt.labels)      # ["production"]
print(prompt.config)      # Config dict
print(prompt.updated_at)  # ISO timestamp

# Compile with variables
compiled = prompt.compile(variables={"topic": "quantum computing"})
messages = prompt.compile_messages(variables={"topic": "quantum computing"})
```

```python
# Fetch cached prompt (returns frozen CachedPrompt dataclass)
cached = neatlogs.fetch_prompt(name="research-agent", label="production")
```

### Creating and Managing Prompts

```python
# Create — `prompt=` is the template text, `labels=` is required
new_prompt = neatlogs.create_prompt(
    name="research-agent",
    prompt="You are a research assistant for {{topic}}.",
    labels=["staging"],
)

# Move labels to a specific version (does NOT update content)
neatlogs.update_prompt(name="research-agent", version=1, new_labels=["production"])

# Save new content as a new version
neatlogs.save_as_version(
    prompt_name="research-agent",
    content="Updated research assistant for {{topic}}.",
    labels=["staging"],
)

# List all prompts
all_prompts = neatlogs.list_prompts()

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
    print(f"API error: {e}")
except PromptClientError as e:
    print(f"Client error: {e}")
```

### When to Use

- Managing prompts centrally across environments (dev, staging, production)
- A/B testing prompt versions via labels
- Sharing prompts between team members
- Version-controlling prompts server-side

> **Note**: The Prompt Management API is fully implemented in sdk-v3 (`neatlogs/prompt/client.py`). It requires a NeatLogs backend connection with a valid API key. Without one, these functions will raise `PromptApiError`.
