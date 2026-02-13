# Neatlogs Python SDK

A comprehensive LLM tracking system that automatically captures and logs all LLM API calls with detailed metrics.

Auto-instruments LLM calls, frameworks, and custom code with just 6 exports.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/neatlogs.svg)](https://badge.fury.io/py/neatlogs)

## Features

- **Auto-Instrumentation**: Automatically tracks OpenAI, Anthropic, LangChain, LlamaIndex, CrewAI, Haystack, and more
- **Rich Metrics**: Token usage, costs, latency, streaming metrics, and custom attributes
- **Multi-Provider**: OpenAI, Anthropic, Google Gemini, Azure OpenAI, Cohere, Groq, Together and 20+ more
- **Simple API**: Just 6 exports - `init()`, `flush()`, `shutdown()`, `@span()`, `trace()`, `PromptTemplate`
- **Zero Config**: Works out-of-the-box with frameworks (LangChain, CrewAI, LlamaIndex, Haystack)
- **OpenTelemetry Native**: Built on OpenTelemetry + OpenInference standards
- **Session-Aware**: Track multi-turn conversations with automatic session grouping
- **Prompt Versioning**: Track prompt templates, variables, and versions

## Installation

```bash
pip install neatlogs
```

## Quick Start

### 1. Framework Code (Auto-Instrumented)

Just `init()` and your framework code is automatically tracked:

```python
from neatlogs import init, flush, shutdown
from langchain.chains import LLMChain
from openai import OpenAI

# Initialize (that's all you need!)
init(
    api_key="your-api-key",
    endpoint="https://api.neatlogs.com/v4/batch",
    instrumentations=["langchain", "openai"],
)

# Your code works normally - fully auto-instrumented!
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is AI?"}]
)

flush()
shutdown()
```

### 2. Custom Code with `@span()`

Instrument your custom orchestration functions:

```python
from neatlogs import init, span, trace, PromptTemplate

init(api_key="...", instrumentations=["openai"])

# Define prompt template for versioning
template = PromptTemplate([
    {"role": "user", "content": "{{question}}"}
])

@span(kind="RETRIEVER", name="vector_search")
def retrieve_docs(query: str):
    """Your custom retrieval logic."""
    return vector_db.search(query)

@span(kind="AGENT", role="Assistant", goal="Answer questions")
def answer_agent(question: str):
    """Agent with prompt tracking."""
    # Track prompt template inside the function that uses it
    with trace(prompt_template=template):
        messages = template.compile(question=question)
        response = llm.create(messages=messages)
        return response.choices[0].message.content

@span(kind="WORKFLOW", name="qa_workflow")
def qa_workflow(question: str):
    """Top-level workflow orchestration."""
    docs = retrieve_docs(question)
    answer = answer_agent(question)
    return answer

# Just call your workflow - no extra wrapper needed!
result = qa_workflow("What is quantum computing?")

flush()
shutdown()
```

## API Reference

### Core Functions

#### `init()`
Initialize the SDK (call once at startup).

```python
init(
    api_key="your-api-key",                    # Required: your Neatlogs API key
    endpoint="https://api.neatlogs.com/v4/batch",  # Optional: custom endpoint
    instrumentations=["openai", "langchain"],  # Auto-instrument frameworks
    metadata={"env": "production"}             # Optional: global metadata
)
```

**Supported instrumentations**: `openai`, `anthropic`, `langchain`, `llama-index`, `crewai`, `haystack`, `google-genai`, `mcp`

#### `flush()`
Send all pending spans to the server.

```python
flush()
```

#### `shutdown()`
Gracefully shutdown the SDK (call at app exit).

```python
shutdown()
```

---

### Custom Instrumentation

#### `@span(kind=...)`
The **only decorator** you need.

```python
@span(
    kind="WORKFLOW",           # Required: WORKFLOW, AGENT, CHAIN, TOOL, RETRIEVER, EMBEDDING, MCP_TOOL
    name="custom_function",    # Optional: span name (defaults to function name)
    
    # Optional: add custom attributes
    role="Assistant",          # For AGENT: agent role
    goal="Answer questions",   # For AGENT: agent goal
    model="gpt-4o",            # For LLM/EMBEDDING: model name
    tool_name="search_api",    # For TOOL/MCP_TOOL: tool name
)
def my_function():
    pass
```

**Supported span kinds**:
- `WORKFLOW`: Top-level orchestration workflows
- `AGENT`: AI agents (agentic behavior)
- `CHAIN`: Sequential or conditional chains
- `TOOL`: Tool/function calls
- `RETRIEVER`: Vector search, document retrieval
- `EMBEDDING`: Embedding generation
- `MCP_TOOL`: Model Context Protocol tools

#### `trace()`
Context manager for:
1. **Prompt tracking** - Track prompt templates inside LLM calls
2. **Session management** - Group multi-turn conversations
3. **Grouping top-level operations** - If you have multiple workflows in `main()`

```python
# Use case 1: Track prompts
@span(kind="AGENT")
def answer_question(q: str):
    template = PromptTemplate([{"role": "user", "content": "{{q}}"}])
    with trace(prompt_template=template):
        messages = template.compile(q=q)
        response = llm.create(messages=messages)
        return response

# Use case 2: Group operations in main()
def main():
    with trace(name="batch_processor", session_id="batch-123"):
        workflow_1()
        workflow_2()

# Use case 3: Multi-turn session tracking
with trace(session_id="user-456", thread_id="conversation-1"):
    for message in conversation:
        agent_workflow(message)
```

#### `PromptTemplate`
Tracks prompt templates, variables and versions.

```python
template = PromptTemplate([
    {"role": "system", "content": "You are a {{role}}."},
    {"role": "user", "content": "{{question}}"}
])

# Compile with variables
messages = template.compile(role="assistant", question="What is AI?")

# Use with trace() to log prompt
with trace(prompt_template=template):
    response = llm.create(messages=messages)
```

---

## Supported Frameworks

### Auto-Instrumented

- **LangChain** - Chains, agents, tools, retrievers, LLMs
- **LlamaIndex** - Queries, retrievals, agents, tools
- **CrewAI** - Agents, tasks, crews, tools
- **Haystack** - Pipelines, components, retrievers
- **OpenAI** - Chat, completions, embeddings, streaming
- **Anthropic** - Claude chat, streaming
- **Google GenAI** - Gemini models
- **Cohere** - Chat, embeddings
- **Model Context Protocol (MCP)** - MCP tools and servers

### Supported LLM Providers

OpenAI • Anthropic • Google Gemini • Azure OpenAI • Cohere • Groq • Together • Anyscale • Perplexity • Mistral • AWS Bedrock • Replicate • HuggingFace • Ollama • LiteLLM • and 20+ more

---

## Common Patterns

### Pattern 1: Pure Framework Code
```python
from neatlogs import init, flush, shutdown
from langchain.chains import LLMChain

init(api_key="...", instrumentations=["langchain", "openai"])

# Your existing code - zero changes!
chain = LLMChain(...)
result = chain.run("query")

flush()
shutdown()
```

### Pattern 2: Custom Workflow with Prompt Tracking
```python
from neatlogs import init, span, trace, PromptTemplate

init(api_key="...", instrumentations=["openai"])

template = PromptTemplate([{"role": "user", "content": "{{q}}"}])

@span(kind="AGENT", role="QA Agent")
def answer_question(q: str):
    with trace(prompt_template=template):
        messages = template.compile(q=q)
        response = llm.create(messages=messages)
        return response

@span(kind="WORKFLOW")
def qa_workflow(q: str):
    return answer_question(q)

result = qa_workflow("What is AI?")
flush()
shutdown()
```

### Pattern 3: RAG Pipeline
```python
from neatlogs import init, span, trace, PromptTemplate

init(api_key="...", instrumentations=["openai"])

@span(kind="RETRIEVER", name="vector_search")
def retrieve_docs(query: str):
    return vector_db.search(query, top_k=5)

@span(kind="TOOL", tool_name="rerank")
def rerank_docs(docs, query: str):
    return reranker.rerank(docs, query)

@span(kind="AGENT", role="RAG Agent")
def generate_answer(query: str, docs):
    template = PromptTemplate([
        {"role": "user", "content": "Context: {{context}}\nQuestion: {{query}}"}
    ])
    with trace(prompt_template=template):
        context = "\n".join([d.content for d in docs])
        messages = template.compile(context=context, query=query)
        return llm.create(messages=messages)

@span(kind="WORKFLOW", name="rag_pipeline")
def rag_pipeline(query: str):
    docs = retrieve_docs(query)
    ranked = rerank_docs(docs, query)
    answer = generate_answer(query, ranked)
    return answer

result = rag_pipeline("What is quantum computing?")
flush()
shutdown()
```

### Pattern 4: Multi-Turn Conversation
```python
from neatlogs import init, span, trace

# Enable auto_session for automatic session management
init(api_key="...", instrumentations=["openai"], auto_session=True)

@span(kind="AGENT", role="Chat Assistant")
def chat(message: str, history: list):
    messages = history + [{"role": "user", "content": message}]
    return llm.create(messages=messages)

# Group entire conversation with session tracking
with trace(session_id="user-123", thread_id="chat-456"):
    history = []
    for user_message in conversation:
        response = chat(user_message, history)
        history.append({"role": "assistant", "content": response})

flush()
shutdown()
```


## Configuration

### Environment Variables
```bash
NEATLOGS_API_KEY=your-api-key
NEATLOGS_ENDPOINT=https://api.neatlogs.com/v4/batch
```

### Initialization Options
```python
init(
    api_key="...",                                  # Required: API key
    endpoint="https://api.neatlogs.com/v4/batch",  # Optional: endpoint
    instrumentations=["openai", "langchain"],       #  Optional: project ID
    workflow_name="my-workflow",                    # Optional: workflow name
)
```

---

## Best Practices

1. **Use auto-instrumentation when possible** - Just `init()` and you're done
2. **`@span()` for custom orchestration** - Wrap your custom workflow, agent, and tool functions
3. **`trace()` for prompts** - Track prompt templates inside functions that use LLMs
4. **`trace()` for sessions** - Group multi-turn conversations with `session_id` and `thread_id`
5. **Always `flush()` and `shutdown()`** - Ensure all spans are sent before exit

---

## Examples

See the `/examples` directory for 60+ comprehensive examples:
- Framework examples (LangChain, CrewAI, LlamaIndex, Haystack)
- Provider examples (OpenAI, Anthropic, Google, Cohere)
- Pattern examples (RAG, agents, tools, streaming, async)
- Guardrail integration examples

---

## License

MIT License - see LICENSE file for details
