# Neatlogs SDK v4 Examples

This directory contains comprehensive examples demonstrating Neatlogs SDK v4 with various AI frameworks and libraries.

## Overview

All examples showcase:
- ✅ **Dual instrumentation** (OpenInference + OpenLLMetry)
- ✅ **Smart attribute merging** (deduplicated, complementary data)
- ✅ **Context propagation** (HTTP spans as children of LLM/TOOL spans)
- ✅ **OpenInference span kinds** (LLM, AGENT, TOOL, RETRIEVER, EMBEDDING, CHAIN, etc.)
- ✅ **Cost tracking** (automatic calculation from pricing.json)
- ✅ **Explicit prompt capture** (no AST magic)

## Examples

### Basic Usage

**01_minimal_usage.py** - Zero-config tracing
- Simplest possible setup
- Auto-instrumentation of OpenAI
- Automatic token counts and cost calculation

**02_prompt_variable_capture.py** - Prompt variable capture
- Using `@observe()` decorator for auto-capture
- Using `with trace()` context manager for explicit tracking
- Prompt template and variable propagation to LLM spans
- Version management

### Agent Frameworks

**03_langchain_agent.py** - LangChain ReAct agent
- Agent with custom tools
- Tool execution tracing
- HTTP calls as children of tool spans

**04_langgraph_agent.py** - LangGraph state machine
- State-based agent workflows
- Node execution tracing
- Conditional edges and transitions

**05_crewai_multi_agent.py** - CrewAI multi-agent system
- Multiple agents collaborating
- Sequential and parallel task execution
- Per-agent metrics and costs

### Direct SDK Usage

**06_openai_direct.py** - OpenAI SDK features
- Basic chat completions
- Streaming responses (time-to-first-token)
- Function calling / tools
- Embeddings generation

**07_anthropic_claude.py** - Anthropic Claude SDK
- Basic chat with Claude
- Prompt caching (cache metrics tracked!)
- Tool use / function calling
- Vision capabilities

**08_dspy_programs.py** - DSPy programs
- Chain of Thought reasoning
- Multi-hop reasoning patterns
- Prompt optimization tracing
- Module composition

### RAG & Vector Databases

**09_rag_with_chromadb.py** - RAG with ChromaDB
- Document embedding and indexing
- Vector similarity search (RETRIEVER spans)
- Context-aware answer generation
- Full RAG pipeline tracing

## Running Examples

### Prerequisites

```bash
# Install dependencies
pip install openai anthropic langchain langgraph crewai dspy-ai chromadb

# Install Neatlogs SDK
cd /Users/tanishabanik/Projects/neatlogs
pip install -e .

# Set API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export NEATLOGS_API_KEY="your-neatlogs-key"
```

### Run an example

```bash
cd /Users/tanishabanik/Projects/neatlogs/neatlogs/sdk/neatlogs_sdk_v4_langfuse/examples

# Run minimal example
python 01_minimal_usage.py

# Run LangChain agent
python 03_langchain_agent.py

# Run RAG example
python 09_rag_with_chromadb.py
```

## What Gets Traced

Every example demonstrates complete observability:

### Span Hierarchy
```
CHAIN (workflow)
├─ AGENT (reasoning agent)
│  └─ LLM (agent thinking)
│     └─ HTTP (API call) ← Correctly parented!
├─ TOOL (function execution)
│  └─ HTTP (tool makes HTTP call) ← Correctly parented!
├─ RETRIEVER (vector search)
│  ├─ EMBEDDING (query embedding)
│  │  └─ HTTP (OpenAI embeddings API)
│  └─ Database query spans
└─ LLM (final answer generation)
   └─ HTTP (OpenAI chat API)
```

### Attributes Collected

From **OpenInference**:
- `openinference.span.kind` (LLM, AGENT, TOOL, RETRIEVER, etc.)
- `llm.cost.total`, `llm.cost.prompt`, `llm.cost.completion`
- `llm.prompt_template`, `llm.prompt_template_variables`
- `embedding.embeddings` (vector data)
- `retrieval.documents` (retrieved docs)

From **OpenLLMetry**:
- `llm.is_streaming` (streaming indicator)
- `llm.response.finish_reason` (stop reason)
- `traceloop.entity.name`, `traceloop.entity.input`, `traceloop.entity.output`
- Streaming metrics (time to first token)

**Merged** (deduplicated):
- Token counts (from both, canonicalized to OpenInference format)
- Model name, temperature, top_p (mapped to OpenInference format)
- Messages (prompt and completion)

## Instrumentation Strategies

### Explicit Libraries (Recommended)

**Best for most use cases** - only instruments what you actually use:

```python
init(
    instrumentations=["openai", "langchain"],
    # Clean output, only checks installed libraries you specify
)
```

### Tag-Based Instrumentation

**Use when exploring** or when you want to instrument all libraries of a type:

```python
init(
    instrument_tags=["llm", "agent", "retrieval"],
    # Attempts to instrument ALL libraries matching these tags
    # May check 20+ libraries even if not installed
)
```

**Available tags:**
- `llm` → openai, anthropic, cohere, bedrock, groq, vertexai, etc. (~15 providers)
- `embedding` → openai, cohere, huggingface
- `retrieval` → chromadb, pinecone, weaviate, qdrant, milvus
- `agent` → langchain, llamaindex, crewai, autogen
- `tool` → langchain, llamaindex
- `http` → requests, httpx, urllib3, aiohttp (always recommended)

**Note:** Tag-based instrumentation checks all libraries in the tag, which may create verbose output. Use explicit library names for cleaner logs.

## Troubleshooting

### HTTP spans not appearing as children?

Make sure `enable_http_tracing=True` (default) and threading instrumentation is enabled (automatic).

### Missing cost data?

- Check if model name matches entries in `pricing.json`
- OpenInference may provide cost directly (preferred)
- Fallback calculation uses pricing.json

### Spans not appearing?

- Verify `api_key` is set correctly
- Check `debug=True` for instrumentation logs
- Ensure libraries are installed (SDK only instruments what's available)

### Library not instrumented?

Add it explicitly:
```python
init(
    instrumentations=["your-library"],
    debug=True  # See what gets instrumented
)
```

## Support Matrix

| Framework | OpenLLMetry | OpenInference | Example |
|-----------|-------------|---------------|---------|
| OpenAI | ✅ | ✅ | 01, 06 |
| Anthropic | ✅ | ✅ | 07 |
| LangChain | ✅ | ✅ | 03 |
| LangGraph | ✅ (via LangChain) | ✅ (via LangChain) | 04 |
| CrewAI | ✅ | ❌ | 05 |
| DSPy | ✅ | ✅ | 08 |
| ChromaDB | ✅ | ✅ | 09 |
| Pinecone | ✅ | ❌ | - |
| Weaviate | ✅ | ✅ | - |
| Qdrant | ✅ | ✅ | - |

## Next Steps

1. **Start simple**: Run `01_minimal_usage.py`
2. **Try your framework**: Find the matching example
3. **Adapt to your code**: Copy patterns from examples
4. **Monitor in production**: Use `sample_rate` for performance

## Questions?

Check the main SDK documentation or open an issue!
