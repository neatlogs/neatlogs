# Neatlogs SDK Test Analysis Report

> **Generated**: January 8, 2025  
> **Purpose**: Document SDK compatibility issues and test failures for Notion documentation

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Test Files | 10 |
| Total Tests | ~150+ |
| Tests Passing | ❌ ~2 |
| Tests Failing | ❌ ~23+ |
| Tests Skipped | ⚠️ ~92 (Agno/DSPy) |

**Key Finding**: The SDK has widespread span creation failures across all providers. Most OpenInference instrumentations are not properly capturing LLM calls.

---

## Test Results by Provider

### ❌ OpenAI (`test_openai_instrumentation.py`)

| Test | Status | Error |
|------|--------|-------|
| `test_basic_chat_completion` | ✅ Pass | - |
| `test_openai_streaming_attributes` | ❌ Fail | BUG: Output not captured correctly |
| `test_openai_tool_calling_span_creation` | ❌ Fail | IndexError: tuple index out of range |
| `test_openai_complex_content_array` | ❌ Fail | IndexError: tuple index out of range |
| `test_openai_extra_metadata_capture` | ❌ Fail | IndexError: tuple index out of range |

**SDK Issues**:
- Streaming response chunks not aggregated to `output.value`
- Tool calls with `content: null` crash the instrumentation
- Multi-part content arrays not flattened properly
- Custom metadata in `extra_body` not captured

---

### ❌ Anthropic (`test_anthropic_instrumentation.py`)

| Test | Status | Error |
|------|--------|-------|
| All tests | ❌ Fail | `CRITICAL: No span created` |

**SDK Issue**: Instrumentation not hooking into Anthropic API calls.

---

### ❌ Gemini (`test_gemini_instrumentation.py`)

| Test | Status | Error |
|------|--------|-------|
| `test_gemini_generate_content` | ❌ Fail | No span created |
| `test_gemini_streaming` | ❌ Fail | No span created |
| `test_gemini_tool_calling` | ❌ Fail | No span created |
| `test_gemini_vertex_mode` | ❌ Fail | No span created |

**SDK Issue**: Google GenAI instrumentation not creating spans.

---

### ❌ LiteLLM (`test_litellm_instrumentation.py`)

| Test | Status | Error |
|------|--------|-------|
| `test_litellm_basic_completion` | ❌ Fail | No span created |

**SDK Issue**: LiteLLM wrapper not instrumented.

---

### ❌ Google ADK (`test_google_adk_instrumentation.py`)

| Test | Status | Error |
|------|--------|-------|
| All tests | ❌ Fail | No span created |

**SDK Issue**: Google ADK instrumentation not functional.

---

### ❌ OpenAI Agents (`test_openai_agents_instrumentation.py`)

| Test | Status | Error |
|------|--------|-------|
| `test_basic_agent_sync` | ❌ Fail | No span created |

**SDK Issue**: OpenAI Agents SDK not instrumented properly.

---

### ❌ CrewAI (`test_crewai_instrumentation.py`)

| Pattern | Test | Status | Error |
|---------|------|--------|-------|
| Basic Sync | `test_basic_crew_sync` | ❌ Fail | I/O operation on closed file |
| Async | `test_basic_crew_async` | ❌ Fail | I/O operation on closed file |
| Multi-Agent | `test_multi_agent_sequential` | ❌ Fail | Span exporter lifecycle issue |
| Tools | `test_crew_with_tools` | ❌ Fail | - |
| ... | (all 20 tests) | ❌ Fail | - |

**SDK Issue**: Span exporter closes before crew execution completes.

**Version Concern**: `crewai==1.5.0` is pinned (exact version).

---

### ❌ LangChain (`test_langchain_instrumentation.py`)

| Pattern | Test | Status | Error |
|---------|------|--------|-------|
| LCEL Chain | `test_lcel_basic_chain` | ❌ Fail | No LLM spans found |
| Legacy LLMChain | `test_legacy_llmchain` | ❌ Fail | No LLM spans found |
| RetrievalQA | `test_retrieval_qa_chromadb` | ❌ Fail | No LLM spans found |
| Agent + Tools | `test_agent_with_tools` | ❌ Fail | No TOOL spans found |
| Streaming | `test_lcel_streaming` | ❌ Fail | No LLM spans found |
| Batch | `test_batch_processing` | ❌ Fail | No LLM spans found |

**SDK Issue**: LCEL RunnableSequence not being instrumented.

**Version Concern**: LangChain versions may be too new (`>=1.1.2`).

---

### ⏳ LlamaIndex (`test_llamaindex_instrumentation.py`)

**Status**: Newly Added - 20 Patterns  
**Tests**: 20 test cases  
**Patterns Covered**:
- Basic LLM Query
- Async Query Execution
- RAG Pipeline
- ReAct Agent with Tools
- Custom Query Engine
- Vector + Summary Index Combined
- Error Handling
- Chat Engine Conversation
- Streaming Responses
- Document Ingestion Pipeline
- Query Router
- Sub-Question Engine
- Custom Callbacks
- Embeddings
- Retry Logic
- Multi-Index Composed Query
- Response Synthesizers
- Node Postprocessors
- Response Evaluation
- Structured Output (Pydantic)

---

### ⏳ Agno (`test_agno_instrumentation.py`)

**Status**: Existing - Already Comprehensive  
**Tests**: 35 test cases (1341 lines)  
**Patterns Covered**:
- Basic Agent Execution (Sync & Async)
- Multi-Agent Systems (Tic Tac Toe Style)
- Tool-Calling Agents
- Multi-Model Agent Federation
- Streaming Responses
- Knowledge Base Integration (RAG)
- Workflows & State Machines
- Error Handling & Edge Cases

---

### ⏳ DSPy (`test_dspy_instrumentation.py`)

**Status**: Existing - Already Comprehensive  
**Tests**: 57 test cases (995 lines)  
**Patterns Covered**:
- Basic DSPy Modules (ChainOfThought, ReAct)
- Retrieval-Augmented Generation (RAG)
- Multi-hop Reasoning Chains
- DSPy Program Compilation/Optimization
- Custom Signatures & Modules
- Multi-Model DSPy Programs
- Async Execution Patterns
- Streaming Responses
- Evaluation Metrics & Validation
- Error Handling & Edge Cases

---

## Version Compatibility Matrix

| Framework | SDK Version | Instrumentation | Status |
|-----------|-------------|-----------------|--------|
| OpenAI | >=1.0.0 | 0.1.32+ | ⚠️ Issues |
| Anthropic | >=0.75.0 | 0.1.20+ | ❌ Broken |
| Google GenAI | >=1.55.0 | 0.1.8+ | ❌ Broken |
| CrewAI | ==1.5.0 | 0.1.17+ | ⚠️ Lifecycle |
| LangChain | >=1.1.2 | 0.1.56+ | ❌ LCEL issues |
| LiteLLM | >=1.80.11 | 0.1.28+ | ❌ Broken |
| Agno | >=2.3.13 | 0.1.25+ | ⏳ Untested |
| DSPy | >=2.6.13 | 0.1.32+ | ⏳ Untested |

---

## SDK Loopholes (Not Covered by Tests)

1. **Rate Limiting (429)** - No tests for throttled responses
2. **Partial Stream Failure** - Connection drops mid-stream
3. **Context Overflow** - Very large prompts
4. **Concurrent Requests** - Multiple simultaneous calls
5. **Token Counting** - Accuracy validation
6. **Multi-modal** - Image/audio content beyond text
7. **Multi-step Tool Chains** - Tool calling sequences
8. **Memory Cleanup** - Long-running session cleanup

---

## Patterns Covered (When Working)

### CrewAI (20 Patterns)
- ✅ Basic sync/async crew execution
- ✅ Multi-agent sequential tasks
- ✅ Task dependencies (context flow)
- ✅ Parallel task execution
- ✅ Multi-LLM (different providers per agent)
- ✅ Error handling and recovery
- ✅ Custom tools integration
- ✅ Nested crews
- ✅ Memory/state management
- ✅ Agent delegation
- ✅ Complex production workflows
- ✅ Streaming responses
- ✅ Custom LLM configuration
- ✅ Input variable capture
- ✅ Process configuration (seq/parallel)
- ✅ Max iterations/retries
- ✅ Template variables
- ✅ Shared context
- ✅ Output validation

### LangChain (15 Patterns)
- ✅ LCEL chains (RunnableSequence)
- ✅ LCEL branching (conditionals)
- ✅ Legacy LLMChain
- ✅ Sequential chains
- ✅ RetrievalQA with ChromaDB
- ✅ HyDE retrieval
- ✅ Tool-calling agents
- ✅ Multi-tool agents
- ✅ Pydantic output parsers
- ✅ Multi-output parsing
- ✅ Streaming (sync/async)
- ✅ Error handling
- ✅ Custom runnables
- ✅ Batch processing

---

## Recommended Fixes

### Priority 1: Critical

1. **Fix OpenAI streaming** - Aggregate SSE chunks properly
2. **Fix tool call handling** - Handle `content: null` responses
3. **Fix span exporter lifecycle** - Ensure proper cleanup in CrewAI

### Priority 2: High

4. **Update LangChain LCEL support** - Instrument RunnableSequence
5. **Fix Anthropic/Gemini instrumentation** - Verify hooks are applied
6. **Relax CrewAI version** - Change `==1.5.0` to `>=1.5.0,<2.0.0`

### Priority 3: Medium

7. Add rate limiting tests
8. Add concurrent request tests
9. Add multi-modal tests
10. Add memory cleanup tests

---

## How to Run Tests

```bash
# Install all dependencies
uv sync --group dev --all-extras

# Run all tests
uv run pytest tests/ -v

# Run specific provider
uv run pytest tests/test_openai_instrumentation.py -v

# Run with coverage
uv run pytest tests/ --cov=neatlogs --cov-report=html
```

---

## Conclusion

The neatlogs SDK has comprehensive test coverage for patterns, but the underlying OpenInference instrumentations are failing across almost all providers. The primary issues are:

1. **Span creation failures** - Most instrumentations not creating spans
2. **Lifecycle issues** - Span exporters closing prematurely
3. **Response parsing** - Streaming and tool calls not handled correctly
4. **Version compatibility** - Some dependencies may be too new/pinned

Before adding new tests, the existing instrumentation issues need to be resolved at the OpenInference library level or within the neatlogs SDK's integration layer.
