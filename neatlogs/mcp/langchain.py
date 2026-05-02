"""LangChain/LangGraph callback adapter that captures all span kinds into the shared buffer."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from . import context as ctx
from .buffer import SpanBuffer, SpanRecord

# ---------------------------------------------------------------------------
# Class-name detection sets for auto-reclassification
# ---------------------------------------------------------------------------

EMBEDDING_CLASSES = frozenset({
    "OpenAIEmbeddings", "AzureOpenAIEmbeddings", "CohereEmbeddings",
    "HuggingFaceEmbeddings", "HuggingFaceBgeEmbeddings",
    "GoogleGenerativeAIEmbeddings", "VertexAIEmbeddings",
    "BedrockEmbeddings", "OllamaEmbeddings", "MistralAIEmbeddings",
    "VoyageAIEmbeddings", "JinaEmbeddings",
})

RERANKER_CLASSES = frozenset({
    "CohereRerank", "FlashrankRerank", "CrossEncoderReranker",
    "JinaRerank", "RankLLMRerank", "MixedbreadAIReranker",
    "ColbertReranker",
})

GUARDRAIL_CLASSES = frozenset({
    "Guard", "NeMoGuardrails", "GuardrailsOutput",
    "InputGuardrail", "OutputGuardrail",
})

EVALUATOR_CLASSES = frozenset({
    "CriteriaEvalChain", "LabeledCriteriaEvalChain",
    "QAEvalChain", "ContextQAEvalChain",
    "StringEvaluator", "PairwiseStringEvaluator",
})

VECTOR_STORE_CLASSES = frozenset({
    "Pinecone", "Qdrant", "Chroma", "FAISS", "Weaviate",
    "Milvus", "PGVector", "ElasticVectorSearch",
    "ElasticsearchStore", "AzureSearch", "AzureCosmosDBVectorSearch",
    "OpenSearchVectorSearch", "MongoDBAtlasVectorSearch",
    "SupabaseVectorStore", "Redis", "Vectara",
})

AGENT_CLASSES = frozenset({
    "AgentExecutor", "OpenAIFunctionsAgent", "OpenAIAssistant",
    "StructuredChatAgent", "XMLAgent", "ReActAgent",
})

MCP_TOOL_INDICATORS = frozenset({"mcp", "mcp_tool", "MCP"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any, max_len: int = 50_000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    return s[:max_len] if len(s) > max_len else s


def _classify_chain(serialized: Dict[str, Any], parent_run_id: Optional[UUID]) -> str:
    name = serialized.get("name", "") or serialized.get("id", [""])[-1]
    if name in RERANKER_CLASSES:
        return "reranker"
    if name in GUARDRAIL_CLASSES:
        return "guardrail"
    if name in EVALUATOR_CLASSES:
        return "evaluator"
    if name in AGENT_CLASSES:
        return "agent"
    if parent_run_id is None:
        return "workflow"
    return "chain"


def _classify_retriever(serialized: Dict[str, Any]) -> str:
    name = serialized.get("name", "") or serialized.get("id", [""])[-1]
    if name in VECTOR_STORE_CLASSES:
        return "vector_store"
    return "retriever"


def _extract_model_name(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> Optional[str]:
    for path in [
        lambda: serialized.get("kwargs", {}).get("model_name"),
        lambda: serialized.get("kwargs", {}).get("model"),
        lambda: kwargs.get("invocation_params", {}).get("model_name"),
        lambda: kwargs.get("invocation_params", {}).get("model"),
    ]:
        v = path()
        if v:
            return str(v)
    return None


def _extract_provider(model_name: Optional[str], serialized: Dict[str, Any]) -> str:
    name = serialized.get("name", "") or serialized.get("id", [""])[-1]
    name_lower = name.lower()
    if "openai" in name_lower:
        return "openai"
    if "anthropic" in name_lower or "claude" in name_lower:
        return "anthropic"
    if "google" in name_lower or "gemini" in name_lower or "vertexai" in name_lower:
        return "google_genai"
    if "groq" in name_lower:
        return "groq"
    if "mistral" in name_lower:
        return "mistralai"
    if "bedrock" in name_lower:
        return "bedrock"
    if "ollama" in name_lower:
        return "ollama"
    if model_name:
        m = model_name.lower()
        if "gpt" in m or "o1" in m or "o3" in m:
            return "openai"
        if "claude" in m:
            return "anthropic"
        if "gemini" in m:
            return "google_genai"
    return ""


def _extract_messages_attrs(
    messages: List[Any], prefix: str = "neatlogs.llm.input_messages"
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for i, msg in enumerate(messages):
        if hasattr(msg, "type") and hasattr(msg, "content"):
            role = getattr(msg, "type", "unknown")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", str(block)))
                    elif hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                content = "\n".join(parts)
            attrs[f"{prefix}.{i}.role"] = role
            attrs[f"{prefix}.{i}.content"] = str(content)[:50_000]
        elif isinstance(msg, dict):
            attrs[f"{prefix}.{i}.role"] = msg.get("role", "unknown")
            attrs[f"{prefix}.{i}.content"] = str(msg.get("content", ""))[:50_000]
    return attrs


def _extract_invocation_params(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    params = kwargs.get("invocation_params") or serialized.get("kwargs", {})
    if not params:
        return attrs

    param_map = {
        "temperature": "neatlogs.llm.temperature",
        "max_tokens": "neatlogs.llm.max_tokens",
        "max_output_tokens": "neatlogs.llm.max_tokens",
        "top_p": "neatlogs.llm.top_p",
        "top_k": "neatlogs.llm.top_k",
        "frequency_penalty": "neatlogs.llm.frequency_penalty",
        "presence_penalty": "neatlogs.llm.presence_penalty",
        "stop": "neatlogs.llm.stop_sequences",
        "stop_sequences": "neatlogs.llm.stop_sequences",
    }
    for src_key, target_key in param_map.items():
        v = params.get(src_key)
        if v is not None:
            attrs[target_key] = _safe_json(v) if isinstance(v, (list, dict)) else v

    if params.get("stream"):
        attrs["neatlogs.llm.is_streaming"] = True
    if params.get("response_format"):
        attrs["neatlogs.llm.request.structured_output_schema"] = _safe_json(params["response_format"])
    if params.get("reasoning_effort"):
        attrs["neatlogs.llm.reasoning_effort"] = str(params["reasoning_effort"])

    safe_params = {k: v for k, v in params.items() if not k.startswith("_") and k != "api_key"}
    attrs["neatlogs.llm.invocation_parameters"] = _safe_json(safe_params)

    return attrs


class LangChainAdapter:
    """LangChain BaseCallbackHandler that captures spans into a shared SpanBuffer.

    Auto-detects span kinds: WORKFLOW, AGENT, CHAIN, LLM, TOOL, RETRIEVER,
    EMBEDDING, RERANKER, VECTOR_STORE, MCP_TOOL, GUARDRAIL, EVALUATOR.
    """

    def __init__(self, buffer: SpanBuffer, workflow_name: str, framework: str = "langchain"):
        self._buffer = buffer
        self._workflow_name = workflow_name
        self._framework = framework
        self._run_to_span: Dict[str, str] = {}
        self._run_to_trace: Dict[str, str] = {}
        self._llm_start_times: Dict[str, float] = {}
        self._first_token_times: Dict[str, float] = {}
        self._handler: Any = None

    @property
    def handler(self) -> Any:
        if self._handler is None:
            self._handler = self._create_handler()
        return self._handler

    def _create_handler(self) -> Any:
        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            raise ImportError("langchain-core is required: pip install langchain-core")

        adapter = self

        class _Handler(BaseCallbackHandler):
            name = "NeatlogsLangChainCallback"

            # -- Chain --
            def on_chain_start(
                self,
                serialized: Dict[str, Any],
                inputs: Dict[str, Any],
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_chain_start(
                    serialized, inputs, run_id, parent_run_id, tags, metadata, **kwargs
                )

            def on_chain_end(
                self,
                outputs: Dict[str, Any],
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_chain_end(outputs, run_id, parent_run_id)

            def on_chain_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_chain_error(error, run_id)

            # -- LLM --
            def on_chat_model_start(
                self,
                serialized: Dict[str, Any],
                messages: List[List[Any]],
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_llm_start(serialized, run_id, parent_run_id, messages=messages, **kwargs)

            def on_llm_start(
                self,
                serialized: Dict[str, Any],
                prompts: List[str],
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_llm_start(serialized, run_id, parent_run_id, prompts=prompts, **kwargs)

            def on_llm_new_token(
                self,
                token: str,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_llm_new_token(token, run_id)

            def on_llm_end(
                self,
                response: Any,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_llm_end(response, run_id)

            def on_llm_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_llm_error(error, run_id)

            # -- Tool --
            def on_tool_start(
                self,
                serialized: Dict[str, Any],
                input_str: str,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                inputs: Optional[Dict[str, Any]] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_tool_start(serialized, input_str, run_id, parent_run_id, inputs)

            def on_tool_end(
                self,
                output: Any,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_tool_end(output, run_id)

            def on_tool_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_tool_error(error, run_id)

            # -- Retriever --
            def on_retriever_start(
                self,
                serialized: Dict[str, Any],
                query: str,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_retriever_start(serialized, query, run_id, parent_run_id)

            def on_retriever_end(
                self,
                documents: Sequence[Any],
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_retriever_end(documents, run_id)

            def on_retriever_error(
                self,
                error: BaseException,
                *,
                run_id: UUID,
                parent_run_id: Optional[UUID] = None,
                **kwargs: Any,
            ) -> None:
                adapter._on_retriever_error(error, run_id)

        return _Handler()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _resolve_parent(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        if parent_run_id is None:
            return None
        return self._run_to_span.get(str(parent_run_id))

    def _ensure_trace(self, run_id: UUID, parent_run_id: Optional[UUID]) -> str:
        rid = str(run_id)
        if rid in self._run_to_trace:
            return self._run_to_trace[rid]

        if parent_run_id:
            parent_trace = self._run_to_trace.get(str(parent_run_id))
            if parent_trace:
                self._run_to_trace[rid] = parent_trace
                return parent_trace

        trace_id = ctx.get_trace_id() or ctx.generate_trace_id()
        ctx.set_trace_id(trace_id)
        self._run_to_trace[rid] = trace_id
        self._buffer.get_or_create_trace(trace_id, self._workflow_name, self._framework)
        return trace_id

    def _start_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        kind: str,
        attributes: Dict[str, Any],
        is_root: bool = False,
    ) -> str:
        trace_id = self._ensure_trace(run_id, parent_run_id)
        span_id = ctx.generate_span_id()
        self._run_to_span[str(run_id)] = span_id

        parent_sid = self._resolve_parent(parent_run_id)
        attributes["neatlogs.span.kind"] = kind

        span = SpanRecord(
            span_id=span_id,
            parent_span_id=parent_sid,
            name=name,
            kind=kind.upper(),
            start_time=_now_iso(),
            attributes=attributes,
            is_root=is_root,
        )
        self._buffer.add_span(trace_id, span)

        if is_root:
            buf = self._buffer.get_or_create_trace(trace_id, self._workflow_name, self._framework)
            buf.root_span_id = span_id

        return span_id

    def _end_span(
        self,
        run_id: UUID,
        status_code: str = "OK",
        status_message: str = "",
        output_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        rid = str(run_id)
        span_id = self._run_to_span.get(rid)
        trace_id = self._run_to_trace.get(rid)
        if not span_id or not trace_id:
            return

        self._buffer.complete_span(
            trace_id,
            span_id,
            _now_iso(),
            status_code=status_code,
            status_message=status_message,
            output_attrs=output_attrs or {},
        )

    # -----------------------------------------------------------------------
    # Chain callbacks
    # -----------------------------------------------------------------------

    def _on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        kind = _classify_chain(serialized, parent_run_id)
        name = serialized.get("name", "") or serialized.get("id", ["chain"])[-1] or "chain"
        is_root = kind == "workflow"

        attrs: Dict[str, Any] = {}
        if is_root:
            attrs["neatlogs.framework"] = self._framework
            if metadata:
                for k, v in metadata.items():
                    attrs[f"neatlogs.user.{k}"] = _safe_json(v)

        if kind not in ("workflow",):
            attrs["input.value"] = _safe_json(inputs)

        if kind == "reranker":
            query = inputs.get("query") or inputs.get("input", "")
            attrs["neatlogs.reranker.query"] = str(query)[:1000]
            if "documents" in inputs:
                attrs["neatlogs.reranker.input_documents"] = _safe_json(inputs["documents"])
            if "top_n" in inputs or "top_k" in inputs:
                attrs["neatlogs.reranker.top_k"] = inputs.get("top_n") or inputs.get("top_k")

        self._start_span(run_id, parent_run_id, name, kind, attrs, is_root=is_root)

    def _on_chain_end(
        self, outputs: Dict[str, Any], run_id: UUID, parent_run_id: Optional[UUID]
    ) -> None:
        output_attrs: Dict[str, Any] = {"output.value": _safe_json(outputs)}

        rid = str(run_id)
        trace_id = self._run_to_trace.get(rid)
        span_id = self._run_to_span.get(rid)
        if trace_id and span_id:
            buf = self._buffer.get_or_create_trace(trace_id, self._workflow_name)
            span = buf.get_span(span_id)
            if span and span.kind.lower() == "reranker" and "output" in outputs:
                output_attrs["neatlogs.reranker.output_documents"] = _safe_json(outputs["output"])

        self._end_span(run_id, output_attrs=output_attrs)

    def _on_chain_error(self, error: BaseException, run_id: UUID) -> None:
        self._end_span(run_id, status_code="ERROR", status_message=str(error))

    # -----------------------------------------------------------------------
    # LLM callbacks
    # -----------------------------------------------------------------------

    def _on_llm_start(
        self,
        serialized: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID],
        *,
        messages: Optional[List[List[Any]]] = None,
        prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        name_str = serialized.get("name", "") or serialized.get("id", ["llm"])[-1]

        # Detect embedding models masquerading as LLM calls
        if name_str in EMBEDDING_CLASSES or "embed" in name_str.lower():
            kind = "embedding"
            attrs: Dict[str, Any] = {
                "neatlogs.embedding.model_name": _extract_model_name(serialized, kwargs) or name_str,
            }
            if prompts:
                attrs["input.value"] = _safe_json(prompts)
            elif messages:
                flat = [str(m) for batch in messages for m in batch]
                attrs["input.value"] = _safe_json(flat)
            attrs.update(_extract_invocation_params(serialized, kwargs))
            self._start_span(run_id, parent_run_id, name_str, kind, attrs)
            return

        kind = "llm"
        model_name = _extract_model_name(serialized, kwargs)
        provider = _extract_provider(model_name, serialized)

        attrs = {}
        if model_name:
            attrs["neatlogs.llm.model_name"] = model_name
        if provider:
            attrs["neatlogs.llm.provider"] = provider
            attrs["neatlogs.llm.system"] = provider
        attrs["neatlogs.llm.request_type"] = "chat" if messages else "completion"

        # Input messages
        if messages and messages[0]:
            attrs.update(_extract_messages_attrs(messages[0]))
        elif prompts:
            for i, p in enumerate(prompts):
                attrs[f"neatlogs.llm.input_messages.{i}.role"] = "user"
                attrs[f"neatlogs.llm.input_messages.{i}.content"] = str(p)[:50_000]

        # Invocation parameters
        attrs.update(_extract_invocation_params(serialized, kwargs))

        # Prompt template from contextvar (set by PromptTemplate.compile())
        pt = ctx.get_prompt_template()
        if pt:
            attrs["neatlogs.llm.prompt_template"] = pt
            pv = ctx.get_prompt_variables()
            if pv:
                attrs["neatlogs.llm.prompt_template_variables"] = _safe_json(pv)

        upt = ctx.get_user_prompt_template()
        if upt:
            attrs["neatlogs.llm.user_prompt_template"] = upt
            upv = ctx.get_user_prompt_variables()
            if upv:
                attrs["neatlogs.llm.user_prompt_template_variables"] = _safe_json(upv)

        # Structured output schema
        response_format = (
            serialized.get("kwargs", {}).get("response_format")
            or kwargs.get("invocation_params", {}).get("response_format")
        )
        if response_format:
            attrs["neatlogs.llm.request.structured_output_schema"] = _safe_json(response_format)

        self._start_span(run_id, parent_run_id, model_name or name_str, kind, attrs)
        self._llm_start_times[str(run_id)] = time.monotonic()

    def _on_llm_new_token(self, token: str, run_id: UUID) -> None:
        rid = str(run_id)
        if rid not in self._first_token_times:
            self._first_token_times[rid] = time.monotonic()

    def _on_llm_end(self, response: Any, run_id: UUID) -> None:
        rid = str(run_id)
        output_attrs: Dict[str, Any] = {}

        # Token usage
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}

        if token_usage:
            # Standard tokens
            if "prompt_tokens" in token_usage or "input_tokens" in token_usage:
                output_attrs["neatlogs.llm.token_count.prompt"] = (
                    token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
                )
            if "completion_tokens" in token_usage or "output_tokens" in token_usage:
                output_attrs["neatlogs.llm.token_count.completion"] = (
                    token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
                )
            if "total_tokens" in token_usage:
                output_attrs["neatlogs.llm.token_count.total"] = token_usage["total_tokens"]

            # Reasoning tokens (OpenAI)
            cdetails = token_usage.get("completion_tokens_details") or {}
            if isinstance(cdetails, dict) and cdetails.get("reasoning_tokens"):
                output_attrs["neatlogs.llm.token_count.reasoning"] = cdetails["reasoning_tokens"]

            # Cache tokens (OpenAI)
            pdetails = token_usage.get("prompt_tokens_details") or {}
            if isinstance(pdetails, dict) and pdetails.get("cached_tokens"):
                output_attrs["neatlogs.llm.token_count.cache_read"] = pdetails["cached_tokens"]

            # Cache tokens (Anthropic)
            if token_usage.get("cache_read_input_tokens"):
                output_attrs["neatlogs.llm.token_count.cache_read"] = token_usage["cache_read_input_tokens"]
            if token_usage.get("cache_creation_input_tokens"):
                output_attrs["neatlogs.llm.token_count.cache_write"] = token_usage["cache_creation_input_tokens"]

        # Output messages + finish reason
        generations = getattr(response, "generations", None) or []
        if generations and generations[0]:
            for i, gen in enumerate(generations[0]):
                message = getattr(gen, "message", None)
                if message:
                    role = getattr(message, "type", "assistant")
                    if role == "ai":
                        role = "assistant"
                    content = getattr(message, "content", "")

                    # Handle content blocks (Anthropic thinking, tool_use, etc.)
                    if isinstance(content, list):
                        text_parts = []
                        thinking_index = len(generations[0])
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "thinking":
                                    output_attrs[f"neatlogs.llm.output_messages.{thinking_index}.role"] = "thinking"
                                    output_attrs[f"neatlogs.llm.output_messages.{thinking_index}.content"] = str(block.get("thinking", ""))[:50_000]
                                    thinking_index += 1
                                elif block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "tool_use":
                                    output_attrs.setdefault("_tool_calls", []).append(block)
                            elif hasattr(block, "type"):
                                btype = getattr(block, "type", "")
                                if btype == "thinking":
                                    output_attrs[f"neatlogs.llm.output_messages.{thinking_index}.role"] = "thinking"
                                    output_attrs[f"neatlogs.llm.output_messages.{thinking_index}.content"] = str(getattr(block, "thinking", ""))[:50_000]
                                    thinking_index += 1
                                elif btype == "text":
                                    text_parts.append(getattr(block, "text", ""))
                        content = "\n".join(text_parts)

                    output_attrs[f"neatlogs.llm.output_messages.{i}.role"] = role
                    output_attrs[f"neatlogs.llm.output_messages.{i}.content"] = str(content)[:50_000]

                    # Tool calls
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        for ti, tc in enumerate(tool_calls):
                            if isinstance(tc, dict):
                                output_attrs[f"neatlogs.llm.tool_calls.{ti}"] = _safe_json(tc)
                            else:
                                output_attrs[f"neatlogs.llm.tool_calls.{ti}"] = _safe_json({
                                    "name": getattr(tc, "name", ""),
                                    "args": getattr(tc, "args", {}),
                                    "id": getattr(tc, "id", ""),
                                })
                else:
                    text = getattr(gen, "text", "")
                    output_attrs[f"neatlogs.llm.output_messages.{i}.role"] = "assistant"
                    output_attrs[f"neatlogs.llm.output_messages.{i}.content"] = str(text)[:50_000]

                # Finish reason
                gen_info = getattr(gen, "generation_info", None) or {}
                if isinstance(gen_info, dict):
                    fr = (
                        gen_info.get("finish_reason")
                        or gen_info.get("stop_reason")
                        or gen_info.get("finishReason")
                    )
                    if fr:
                        output_attrs[f"neatlogs.llm.output_messages.{i}.message.finish_reason"] = str(fr)
                        if i == 0:
                            output_attrs["neatlogs.llm.finish_reason"] = str(fr)

        # Clean up internal keys
        output_attrs.pop("_tool_calls", None)

        # Streaming metrics
        start_t = self._llm_start_times.pop(rid, None)
        first_token_t = self._first_token_times.pop(rid, None)
        if start_t and first_token_t:
            output_attrs["neatlogs.llm.ttft_ms"] = round((first_token_t - start_t) * 1000, 2)
        if first_token_t:
            end_t = time.monotonic()
            output_attrs["neatlogs.llm.streaming_time_to_generate_ms"] = round(
                (end_t - first_token_t) * 1000, 2
            )

        self._end_span(run_id, output_attrs=output_attrs)

    def _on_llm_error(self, error: BaseException, run_id: UUID) -> None:
        self._llm_start_times.pop(str(run_id), None)
        self._first_token_times.pop(str(run_id), None)
        self._end_span(run_id, status_code="ERROR", status_message=str(error))

    # -----------------------------------------------------------------------
    # Tool callbacks
    # -----------------------------------------------------------------------

    def _on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        inputs: Optional[Dict[str, Any]],
    ) -> None:
        name = serialized.get("name", "") or "tool"
        description = serialized.get("description", "")

        # Detect MCP tools
        is_mcp = any(ind in name.lower() for ind in MCP_TOOL_INDICATORS)
        metadata = serialized.get("metadata", {})
        if metadata and any(k.startswith("mcp") for k in metadata):
            is_mcp = True

        kind = "mcp_tool" if is_mcp else "tool"
        attrs: Dict[str, Any] = {
            "neatlogs.tool.name": name,
            "neatlogs.tool.input": _safe_json(inputs or input_str),
        }
        if description:
            attrs["neatlogs.tool.description"] = str(description)[:2000]
        if serialized.get("parameters"):
            attrs["neatlogs.tool.parameters"] = _safe_json(serialized["parameters"])

        if is_mcp:
            attrs["neatlogs.mcp.method"] = "tools/call"
            attrs["neatlogs.mcp.request_argument"] = _safe_json(inputs or input_str)[:4000]

        self._start_span(run_id, parent_run_id, name, kind, attrs)

    def _on_tool_end(self, output: Any, run_id: UUID) -> None:
        output_str = _safe_json(output)
        output_attrs: Dict[str, Any] = {"neatlogs.tool.output": output_str}

        rid = str(run_id)
        trace_id = self._run_to_trace.get(rid)
        span_id = self._run_to_span.get(rid)
        if trace_id and span_id:
            buf = self._buffer.get_or_create_trace(trace_id, self._workflow_name)
            span = buf.get_span(span_id)
            if span and span.kind.lower() == "mcp_tool":
                output_attrs["neatlogs.mcp.response_value"] = output_str[:4000]

        self._end_span(run_id, output_attrs=output_attrs)

    def _on_tool_error(self, error: BaseException, run_id: UUID) -> None:
        self._end_span(run_id, status_code="ERROR", status_message=str(error))

    # -----------------------------------------------------------------------
    # Retriever callbacks
    # -----------------------------------------------------------------------

    def _on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        run_id: UUID,
        parent_run_id: Optional[UUID],
    ) -> None:
        kind = _classify_retriever(serialized)
        name = serialized.get("name", "") or serialized.get("id", ["retriever"])[-1] or "retriever"
        attrs: Dict[str, Any] = {"input.value": str(query)[:50_000]}

        if kind == "vector_store":
            attrs["neatlogs.vectordb.retrieval_query"] = str(query)[:50_000]
            index_name = serialized.get("kwargs", {}).get("index_name") or serialized.get("kwargs", {}).get("collection_name")
            if index_name:
                attrs["neatlogs.vectordb.index_name"] = str(index_name)

        self._start_span(run_id, parent_run_id, name, kind, attrs)

    def _on_retriever_end(self, documents: Sequence[Any], run_id: UUID) -> None:
        output_attrs: Dict[str, Any] = {}

        rid = str(run_id)
        trace_id = self._run_to_trace.get(rid)
        span_id = self._run_to_span.get(rid)
        kind = "retriever"
        if trace_id and span_id:
            buf = self._buffer.get_or_create_trace(trace_id, self._workflow_name)
            span = buf.get_span(span_id)
            if span:
                kind = span.kind.lower()

        for i, doc in enumerate(documents):
            content = getattr(doc, "page_content", "") or str(doc)
            doc_id = getattr(doc, "id", None) or getattr(getattr(doc, "metadata", {}), "get", lambda k, d=None: d)("id")
            meta = getattr(doc, "metadata", {})

            prefix = "neatlogs.retriever.documents" if kind == "retriever" else "neatlogs.vectordb.retrieval_documents"
            if kind == "retriever":
                output_attrs[f"{prefix}.{i}.content"] = str(content)[:10_000]
                if doc_id:
                    output_attrs[f"{prefix}.{i}.id"] = str(doc_id)
                if meta:
                    output_attrs[f"{prefix}.{i}.metadata"] = _safe_json(meta)
            else:
                if i == 0:
                    output_attrs[prefix] = _safe_json([
                        {
                            "content": str(getattr(d, "page_content", "") or str(d))[:5000],
                            "id": str(getattr(d, "id", "") or ""),
                            "metadata": _safe_json(getattr(d, "metadata", {})),
                        }
                        for d in documents
                    ])
                break

        output_attrs["output.value"] = _safe_json([
            str(getattr(d, "page_content", "") or str(d))[:2000] for d in documents
        ])

        self._end_span(run_id, output_attrs=output_attrs)

    def _on_retriever_error(self, error: BaseException, run_id: UUID) -> None:
        self._end_span(run_id, status_code="ERROR", status_message=str(error))
