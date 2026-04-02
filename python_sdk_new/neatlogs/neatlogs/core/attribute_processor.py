import json
import logging
import re
import ast
from typing import Any, Dict, List, Optional

from opentelemetry import metrics
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind

from .logger import get_logger
from .instrumentation_scope_parser import enrich_with_scope_detection

# Matches Python object repr strings like:
#   <function BaseTool.<lambda> at 0x110107be0>
#   <bound method Foo.bar at 0x7f...>
#   <MyClass object at 0x...>
_PYTHON_REPR_RE = re.compile(r'^<[A-Za-z_].*?\bat\s+0x[0-9a-fA-F]+>$')

# Maps neatlogs.provider values → neatlogs.llm.system
# System values are from OpenInferenceLLMSystemValues (openai/anthropic/cohere/mistralai/vertexai)
# For providers not in OI's LLMSystemValues we use the provider name as the system value.
# "aws"/"bedrock" are intentionally omitted: on Bedrock, llm.system is set to the underlying
# model vendor (e.g. "anthropic" for Claude) which is more informative than "aws".
_PROVIDER_TO_SYSTEM: Dict[str, str] = {
    "openai": "openai",
    "azure": "openai",       # Azure OpenAI uses OpenAI's system
    "azure_openai": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "mistral": "mistralai",  # OI LLMSystemValues.MISTRALAI = "mistralai"
    "mistralai": "mistralai",
    "google": "google",
    "vertex_ai": "vertexai", # OI LLMSystemValues.VERTEXAI = "vertexai"
    "groq": "groq",
    "xai": "xai",
    "deepseek": "deepseek",
}


def _is_python_repr(s: str) -> bool:
    return bool(_PYTHON_REPR_RE.match(s.strip()))


def _clean_python_reprs(obj: Any) -> Any:
    """Recursively remove Python object repr strings from a parsed JSON structure."""
    if isinstance(obj, dict):
        return {
            k: _clean_python_reprs(v)
            for k, v in obj.items()
            if not (isinstance(v, str) and _is_python_repr(v))
        }
    if isinstance(obj, list):
        return [
            _clean_python_reprs(item)
            for item in obj
            if not (isinstance(item, str) and _is_python_repr(item))
        ]
    return obj


class UnifiedAttributeProcessor:

    def __init__(
        self,
        mapping_config: Dict[str, Any],
        debug: bool = False,
    ):
        self.mapping = mapping_config
        self.debug = debug
        self.logger = get_logger()
        
        # Enable debug logging if debug mode is on
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
            for handler in self.logger.handlers:
                handler.setLevel(logging.DEBUG)

        self.meter = metrics.get_meter("neatlogs.sdk")

    def _sanitize_io_value(self, val: Any) -> Any:
        """Remove Python object reprs from input.value / output.value JSON strings.

        Also drops the top-level "self" key that CrewAI injects when it serializes
        the entire tool instance as part of the span input.
        """
        if not isinstance(val, str):
            return val
        try:
            parsed = json.loads(val)
            cleaned = _clean_python_reprs(parsed)
            if isinstance(cleaned, dict):
                cleaned.pop("self", None)
            if cleaned != parsed:
                return json.dumps(cleaned)
        except Exception:
            pass
        return val

    def process(self, span: ReadableSpan) -> Dict[str, Any]:
        res_attrs = dict(span.resource.attributes) if span.resource else {}
        attrs = {**res_attrs, **dict(span.attributes)}
        
        # Add span name for downstream processing
        attrs["_span_name"] = span.name
        
        # Extract framework/platform/provider from instrumentation scope
        scope_name = span.instrumentation_scope.name if span.instrumentation_scope else None
        # TODO: In future, track parent spans to get parent_scope_name for better framework detection
        enrich_with_scope_detection(attrs, scope_name, parent_scope_name=None)
        if self.debug:
            trace_id = f"{span.context.trace_id:032x}" if span.context else ""
            span_id = f"{span.context.span_id:016x}" if span.context else ""
            has_crewai_attrs = any(k.startswith("crewai.") for k in attrs.keys())
            self.logger.debug(
                "[ScopeDetection] trace_id=%s span_id=%s span_name=%s scope=%s framework=%s provider=%s platform=%s has_crewai_attrs=%s",
                trace_id,
                span_id,
                span.name,
                scope_name,
                attrs.get("neatlogs.framework"),
                attrs.get("neatlogs.provider"),
                attrs.get("neatlogs.platform"),
                has_crewai_attrs,
            )

        attrs = self._normalize_conventions(span, attrs)

        computed_metrics = self._extract_operational_metrics(span, attrs)
        attrs.update(computed_metrics)

        event_attrs = self._upcycle_events(span)
        attrs.update(event_attrs)

        try:
            from ..config import enrich_invocation_parameters

            enrich_invocation_parameters(attrs, enable_enrichment=True)
        except Exception as e:
            self.logger.warning(f"Failed to enrich invocation parameters: {e}")

        unified = self._apply_namespace_mapping(attrs)
        self._add_intermediate_steps(unified)
        
        # 🔥 CRITICAL: Filter out massive embedding vectors before export
        # Embedding vectors can be 4MB+ (1000 embeddings × 4096 dimensions), causing:
        # - Kafka message size limits (1MB default)
        # - ClickHouse memory exhaustion (6GB+ RAM usage)
        # - Network/storage costs
        span_kind = (unified.get("neatlogs.span.kind") or "").lower()
        if span_kind in ("embedding", "vector_store"):
            unified = self._filter_embedding_vectors(unified)
        
        return unified

    def _normalize_conventions(self, span: ReadableSpan, attrs: Dict[str, Any]) -> Dict[str, Any]:
        if span.kind == SpanKind.CLIENT and self._looks_like_http(attrs):
            attrs["openinference.span.kind"] = "HTTP"

        if "openinference.span.kind" not in attrs and any(
            k.startswith("crewai.crew.") for k in attrs.keys()
        ):
            attrs["openinference.span.kind"] = "CHAIN"

        self._add_crewai_token_usage_fallback(attrs)
        self._add_reasoning_tokens_from_output_value(attrs)
        self._add_crewai_kickoff_telemetry(attrs)
        
        # Extract tool calls from output messages (OpenInference format)
        tool_calls: Dict[int, Dict[str, Any]] = {}
        oi_tool_re = re.compile(
            r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.function\.(name|arguments)$"
        )
        oi_tool_id_re = re.compile(
            r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.id$"
        )

        keys_to_remove: List[str] = []
        for k, v in list(attrs.items()):
            # OpenInference tool call function (name, arguments)
            m = oi_tool_re.match(k)
            if m:
                _msg_idx, call_idx, field = m.groups()
                idx = int(call_idx)
                tool_calls.setdefault(idx, {})[field] = v
                keys_to_remove.append(k)
                continue

            # OpenInference tool call id (separate pattern)
            m = oi_tool_id_re.match(k)
            if m:
                _msg_idx, call_idx = m.groups()
                idx = int(call_idx)
                tool_calls.setdefault(idx, {})["id"] = v
                keys_to_remove.append(k)
                continue

        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            if "id" in tc:
                attrs[f"llm.tool_calls.{idx}.id"] = tc["id"]
            if "name" in tc:
                attrs[f"llm.tool_calls.{idx}.name"] = tc["name"]
            if "arguments" in tc:
                attrs[f"llm.tool_calls.{idx}.arguments"] = tc["arguments"]

        for k in keys_to_remove:
            attrs.pop(k, None)
        
        # Extract tool_call_id and name from input messages (tool response messages)
        input_msg_tool_re = re.compile(
            r"^llm\.input_messages\.(\d+)\.message\.(tool_call_id|name)$"
        )
        for k, v in list(attrs.items()):
            m = input_msg_tool_re.match(k)
            if m:
                msg_idx, field = m.groups()
                # Map to new structured format
                attrs[f"llm.input_messages.{msg_idx}.{field}"] = v
        
        # Extract invalid_tool_calls from output (LangChain AIMessage often includes this)
        # Look for patterns like: llm.output or gen_ai.completion containing invalid_tool_calls
        llm_output = attrs.get("llm.output") or attrs.get("output.value")
        if llm_output and isinstance(llm_output, str):
            try:
                output_data = json.loads(llm_output)
                # Navigate through nested structure to find invalid_tool_calls
                if isinstance(output_data, dict):
                    # LangChain format: generations[0][0].message.invalid_tool_calls
                    generations = output_data.get("generations", [])
                    if generations and len(generations) > 0 and len(generations[0]) > 0:
                        message = generations[0][0].get("message", {})
                        invalid_calls = message.get("invalid_tool_calls", [])
                        if invalid_calls:
                            attrs["llm.invalid_tool_calls"] = json.dumps(invalid_calls)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        
        # Extract tool_call_id from tool.output (for TOOL kind spans)
        # This provides structured field that Kafka consumer currently extracts from JSON
        tool_output = attrs.get("tool.output") or attrs.get("output.value")
        if tool_output and isinstance(tool_output, str):
            try:
                output_data = json.loads(tool_output)
                if isinstance(output_data, dict):
                    tool_call_id = output_data.get("tool_call_id") or output_data.get("toolCallId")
                    if tool_call_id:
                        attrs["tool_call_id"] = tool_call_id
            except (json.JSONDecodeError, TypeError):
                pass

        tool_defs: Dict[int, Dict[str, Any]] = {}
        oi_schema_re = re.compile(r"^llm\.tools\.(\d+)\.tool\.json_schema$")
        ol_fn_re = re.compile(r"^llm\.request\.functions\.(\d+)\.(name|description|input_schema)$")

        keys_to_remove = []
        for k, v in list(attrs.items()):
            m = ol_fn_re.match(k)
            if m:
                idx_s, field = m.groups()
                idx = int(idx_s)
                tool_defs.setdefault(idx, {})[field] = v
                keys_to_remove.append(k)
                continue

            m = oi_schema_re.match(k)
            if m:
                idx = int(m.group(1))
                schema = v
                if isinstance(schema, str):
                    try:
                        schema = json.loads(schema)
                    except Exception:
                        schema = None
                if isinstance(schema, dict):
                    td = tool_defs.setdefault(idx, {})
                    td.setdefault("name", schema.get("name"))
                    td.setdefault("description", schema.get("description"))
                    td.setdefault(
                        "input_schema", schema.get("input_schema") or schema.get("parameters")
                    )
                keys_to_remove.append(k)

        for idx in sorted(tool_defs.keys()):
            td = tool_defs[idx]
            if td.get("name") is not None:
                attrs.setdefault(f"llm.tools.{idx}.name", td["name"])
            if td.get("description") is not None:
                attrs.setdefault(f"llm.tools.{idx}.description", td["description"])
            if td.get("input_schema") is not None:
                val = td["input_schema"]
                if not isinstance(val, str):
                    try:
                        val = json.dumps(val)
                    except Exception:
                        val = str(val)
                attrs.setdefault(f"llm.tools.{idx}.input_schema", val)

        for k in keys_to_remove:
            attrs.pop(k, None)

        if "openinference.span.kind" not in attrs:
            db_system = attrs.get("db.system")
            db_operation = attrs.get("db.operation", "").lower()
          
            if isinstance(db_system, str) and db_system.lower() in {
                "chroma",
                "chromadb",
                "pinecone",
                "qdrant",
                "milvus",
                "marqo",
                "weaviate",
                "lancedb",
                "astra",
                "pgvector",
                "elasticsearch",
            }:
                # Use db.operation (if available) or span name to distinguish RETRIEVER vs VECTOR_STORE
                span_name = attrs.get("_span_name", "").lower()
                
                # Read operations (retrieval)
                retrieval_ops = [
                    "query", "search", "get", "fetch", "find",
                    "retrieve", "scroll", "peek", "discover", "recommend",
                    "aggregate", "hybrid_search", "select"
                ]
                
                # Write operations
                write_ops = [
                    "upsert", "insert", "add", "update", "delete", "create",
                    "drop", "put", "set", "upload", "index"
                ]
                
                # Check db.operation first (most reliable)
                is_retrieval = False
                if db_operation:
                    is_retrieval = any(op in db_operation for op in retrieval_ops)
                    is_write = any(op in db_operation for op in write_ops)
                else:
                    # Fallback to span name
                    is_retrieval = any(op in span_name for op in retrieval_ops)
                
                if is_retrieval:
                    attrs["openinference.span.kind"] = "RETRIEVER"
                else:
                    attrs["openinference.span.kind"] = "VECTOR_STORE"

        if "traceloop.entity.input" in attrs:
            try:
                entity_input = (
                    json.loads(attrs["traceloop.entity.input"])
                    if isinstance(attrs["traceloop.entity.input"], str)
                    else attrs["traceloop.entity.input"]
                )
                if isinstance(entity_input, dict):
                    if "method" in entity_input:
                        attrs["mcp.method.name"] = entity_input["method"]
                    if "params" in entity_input:
                        attrs["mcp.request.argument"] = json.dumps(entity_input["params"])
                    if "tool_name" in entity_input:
                        attrs["mcp.tool.name"] = entity_input["tool_name"]
                        if "arguments" in entity_input and isinstance(
                            entity_input["arguments"], dict
                        ):
                            attrs["mcp.tool.arguments"] = json.dumps(entity_input["arguments"])
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        has_mcp_signal = False
        if isinstance(attrs.get("mcp.method.name"), str) and attrs.get("mcp.method.name"):
            has_mcp_signal = True
        if isinstance(attrs.get("mcp.tool.name"), str) and attrs.get("mcp.tool.name"):
            has_mcp_signal = True
        if "mcp.request.argument" in attrs or "mcp.tool.arguments" in attrs:
            has_mcp_signal = True

        if (
            has_mcp_signal
            and "traceloop.entity.output" in attrs
            and "mcp.response.value" not in attrs
        ):
            attrs["mcp.response.value"] = attrs["traceloop.entity.output"]

        if attrs.get("mcp.method.name") == "initialize" and "traceloop.entity.output" in attrs:
            try:
                output = (
                    json.loads(attrs["traceloop.entity.output"])
                    if isinstance(attrs["traceloop.entity.output"], str)
                    else attrs["traceloop.entity.output"]
                )
                if "protocolVersion" in output:
                    attrs["mcp.protocol_version"] = output["protocolVersion"]
                if "serverInfo" in output and isinstance(output["serverInfo"], dict):
                    info = output["serverInfo"]
                    attrs["mcp.server.name"] = info.get("name", "")
                    attrs["mcp.server.version"] = info.get("version", "")
                if "capabilities" in output:
                    attrs["mcp.server.capabilities"] = json.dumps(output["capabilities"])[:2000]
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        if attrs.get("mcp.method.name") == "tools/list" and "traceloop.entity.output" in attrs:
            try:
                output = (
                    json.loads(attrs["traceloop.entity.output"])
                    if isinstance(attrs["traceloop.entity.output"], str)
                    else attrs["traceloop.entity.output"]
                )
                if "tools" in output and isinstance(output["tools"], list):
                    tools = output["tools"]
                    attrs["mcp.tools.count"] = len(tools)
                    tool_names = [t.get("name") for t in tools if "name" in t]
                    attrs["mcp.tools.names"] = json.dumps(tool_names)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        span_kind = attrs.get("openinference.span.kind", "").upper()
        db_system = attrs.get("db.system", "").lower()

        if span_kind == "EMBEDDING":
            embeddings = []
            for key, value in attrs.items():
                if key.startswith("embedding.embeddings.") and key.endswith(".embedding.text"):
                    try:
                        parts = key.split(".")
                        if len(parts) >= 3:
                            index = int(parts[2])
                            embeddings.append({"index": index, "text": value})
                    except (ValueError, IndexError):
                        continue

            if embeddings:
                embeddings.sort(key=lambda x: x["index"])
                attrs["embeddings_data"] = json.dumps(embeddings)

            # Only skip output if it's a REAL embedding operation from OpenLLMetry
            # (has actual embedding attributes, not just user's @span(kind="EMBEDDING"))
            has_embedding_attrs = any(
                k.startswith("embedding.") or k.startswith("gen_ai.embedding")
                for k in attrs.keys()
            )
            
            if has_embedding_attrs:
                attrs["neatlogs._skip_output_value"] = True

        if db_system == "chroma":
            doc_attrs = {}
            for key in [
                "db.chroma.add.ids_count",
                "db.chroma.add.embeddings_count",
                "db.chroma.add.metadatas_count",
                "db.chroma.add.documents_count",
                "db.chroma.upsert.ids_count",
                "db.chroma.upsert.embeddings_count",
                "db.chroma.upsert.metadatas_count",
                "db.chroma.upsert.documents_count",
            ]:
                if key in attrs:
                    doc_attrs[key.split(".")[-1]] = attrs[key]

            # Chroma's n_results is the requested top-k, not guaranteed returned count.
            if "db.chroma.query.n_results" in attrs:
                doc_attrs["requested_top_k"] = attrs["db.chroma.query.n_results"]

            if "db.chroma.query.include" in attrs:
                doc_attrs["include"] = attrs["db.chroma.query.include"]

            if doc_attrs:
                attrs["document_attributes"] = json.dumps(doc_attrs)

        elif db_system == "marqo":
            input_params = {}
            for key in ["marqo.limit", "marqo.hits_count", "marqo.filter"]:
                if key in attrs:
                    input_params[key.split(".")[-1]] = attrs[key]
            if input_params:
                attrs["retrieval_input_params"] = json.dumps(input_params)

            doc_attrs = {}
            for key in ["marqo.document_count", "marqo.items_processed"]:
                if key in attrs:
                    doc_attrs[key.split(".")[-1]] = attrs[key]
            if doc_attrs:
                attrs["document_attributes"] = json.dumps(doc_attrs)

        elif db_system == "qdrant":
            doc_attrs = {}
            if "qdrant.upsert.points_count" in attrs:
                doc_attrs["points_count"] = attrs["qdrant.upsert.points_count"]
            if doc_attrs:
                attrs["document_attributes"] = json.dumps(doc_attrs)

        elif db_system == "milvus":
            doc_attrs = {}
            for key in [
                "db.milvus.insert.data_count",
                "db.milvus.search.data_count",
                "db.milvus.search.limit",
                "db.milvus.search.output_fields_count",
                "db.milvus.search.result_count",
            ]:
                if key in attrs:
                    doc_attrs[key.replace("db.milvus.", "")] = attrs[key]
            if "db.milvus.search.filter" in attrs:
                doc_attrs["search.filter"] = attrs["db.milvus.search.filter"]
            if doc_attrs:
                attrs["document_attributes"] = json.dumps(doc_attrs)

        return attrs

    def _add_crewai_kickoff_telemetry(self, attrs: Dict[str, Any]) -> None:
        """
        Ensure CrewAI kickoff spans carry telemetry directly so downstream consumers
        do not need cross-batch Crew Created → kickoff correlation.
        """
        span_name = str(attrs.get("_span_name") or "")
        if not (span_name.startswith("Crew_") and span_name.endswith(".kickoff")):
            return

        # Fill in computed values only when OpenInference didn't already set them.
        if "crewai_version" not in attrs:
            try:
                import crewai  # type: ignore

                version = getattr(crewai, "__version__", None)
                if version:
                    attrs["crewai_version"] = str(version)
            except Exception:
                pass

        if "crew_number_of_tasks" not in attrs:
            count = self._coerce_collection_count(
                attrs.get("crew_tasks")
            )
            if count is not None:
                attrs["crew_number_of_tasks"] = count

        if "crew_number_of_agents" not in attrs:
            count = self._coerce_collection_count(
                attrs.get("crew_agents")
            )
            if count is not None:
                attrs["crew_number_of_agents"] = count

    def _coerce_collection_count(self, value: Any) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            try:
                return int(value)
            except Exception:
                return None
        if isinstance(value, (list, tuple, set, dict)):
            return len(value)

        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None

            # Direct integer string.
            if re.fullmatch(r"\d+", s):
                try:
                    return int(s)
                except Exception:
                    return None

            # Try JSON first.
            try:
                parsed = json.loads(s)
                if isinstance(parsed, (list, tuple, set, dict)):
                    return len(parsed)
            except Exception:
                pass

            # Fallback for Python-repr style lists from some instrumentations.
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set, dict)):
                    return len(parsed)
            except Exception:
                pass

        return None

    def _add_crewai_token_usage_fallback(self, attrs: Dict[str, Any]) -> None:
        """
        Parse CrewAI aggregate token usage strings like:
          "total_tokens=67305 prompt_tokens=46983 cached_prompt_tokens=0 completion_tokens=20322 successful_requests=27"
        and map them to OpenInference-style token keys:
          - llm.token_count.prompt / completion / total
          - llm.token_count.prompt_details.cache_read (best-effort)
        Only applies when token_count fields are not already present.
        """
        # neatlogs.crew.token_usage: set by our OI CrewAI patch (_patch_openinference_crewai_crew_outputs)
        usage = attrs.get("neatlogs.crew.token_usage")
        if not isinstance(usage, str) or not usage:
            return

        if any(
            k in attrs
            for k in (
                "llm.token_count.prompt",
                "llm.token_count.completion",
                "llm.token_count.total",
            )
        ):
            return

        parsed: Dict[str, int] = {}
        for key, val in re.findall(r"([a-zA-Z_]+)=(\d+)", usage):
            try:
                parsed[key] = int(val)
            except Exception:
                continue

        if "prompt_tokens" in parsed:
            attrs["llm.token_count.prompt"] = parsed["prompt_tokens"]
        if "completion_tokens" in parsed:
            attrs["llm.token_count.completion"] = parsed["completion_tokens"]
        if "total_tokens" in parsed:
            attrs["llm.token_count.total"] = parsed["total_tokens"]

        # CrewAI reports cached_prompt_tokens; treat as cache-read input tokens.
        if "cached_prompt_tokens" in parsed:
            attrs["llm.token_count.prompt_details.cache_read"] = parsed["cached_prompt_tokens"]

    def _add_reasoning_tokens_from_output_value(self, attrs: Dict[str, Any]) -> None:
        """
        Fallback: parse output.value JSON to extract reasoning_tokens when
        llm.token_count.completion_details.reasoning is absent as a span attribute.

        Covers spans where the OpenInference OpenAI/LiteLLM instrumentor fails to
        extract it as a top-level attribute (e.g. Azure deployments via LiteLLM in
        CrewAI), even though the full API response in output.value contains the value.
        """
        if "llm.token_count.completion_details.reasoning" in attrs:
            return
        if "llm.usage.reasoning_tokens" in attrs:
            return

        output_value = attrs.get("output.value")
        if not isinstance(output_value, str):
            return

        try:
            parsed = json.loads(output_value)
            # Standard OpenAI / Azure chat completions response shape
            usage = parsed.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            reasoning = details.get("reasoning_tokens")
            if reasoning and reasoning > 0:
                attrs["llm.token_count.completion_details.reasoning"] = reasoning
        except Exception:
            pass

    def _add_intermediate_steps(self, unified: Dict[str, Any]) -> None:
        """
        Populate `neatlogs.llm.intermediate_steps` when the LLM content includes ReAct-style
        markers (Thought/Context/Action/Action Input/Observation/Final Answer).

        This is intentionally compact and aggressively truncated to avoid payload bloat.
        Full tool outputs remain on tool spans.
        """

        if "neatlogs.llm.intermediate_steps" in unified:
            return
        if str(unified.get("neatlogs.span.kind", "")).lower() != "llm":
            return

        steps = self._extract_react_steps_from_messages(unified)
        if not steps:
            return

        unified["neatlogs.llm.intermediate_steps"] = json.dumps(steps, ensure_ascii=True)

    def _extract_react_steps_from_messages(self, unified: Dict[str, Any]) -> List[Dict[str, str]]:
        def _truncate(val: str, max_len: int) -> str:
            val = (val or "").strip()
            if len(val) <= max_len:
                return val
            return val[:max_len] + f"...(truncated,len={len(val)})"

        output_texts = self._collect_role_texts(
            unified, prefix="neatlogs.llm.output_messages", role="assistant"
        )
        steps = self._parse_react_steps(output_texts)
        if steps:
            return steps

        input_texts = self._collect_role_texts(
            unified, prefix="neatlogs.llm.input_messages", role="assistant"
        )
        return self._parse_react_steps(input_texts)

    def _collect_role_texts(self, unified: Dict[str, Any], prefix: str, role: str) -> List[str]:
        idx_re = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.content$")
        idxs: set[int] = set()
        for k in unified.keys():
            m = idx_re.match(k)
            if m:
                try:
                    idxs.add(int(m.group(1)))
                except Exception:
                    pass

        texts: List[str] = []
        for i in sorted(idxs):
            r = unified.get(f"{prefix}.{i}.role")
            if not isinstance(r, str) or r.lower() != role:
                continue
            c = unified.get(f"{prefix}.{i}.content")
            if isinstance(c, str) and "thought:" in c.lower():
                texts.append(c[:20000])
        return texts

    def _parse_react_steps(self, texts: List[str]) -> List[Dict[str, str]]:
        if not texts:
            return []

        marker_re = re.compile(
            r"(?im)(?:^|\n)\s*(Thought|Context|Action|Action Input|Observation|Final Answer)\s*:\s*"
        )

        def _truncate(val: str, max_len: int) -> str:
            val = (val or "").strip()
            if len(val) <= max_len:
                return val
            return val[:max_len] + f"...(truncated,len={len(val)})"

        all_steps: List[Dict[str, str]] = []

        for text in texts:
            matches = list(marker_re.finditer(text))
            if not matches:
                continue

            cur: Dict[str, str] = {}

            def _commit() -> None:
                nonlocal cur
                if not cur:
                    return
                if not any(v for v in cur.values()):
                    cur = {}
                    return
                if all_steps and all_steps[-1] == cur:
                    cur = {}
                    return
                all_steps.append(cur)
                cur = {}

            for idx, m in enumerate(matches):
                label = m.group(1).strip().lower()
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                value = text[start:end].strip()

                if label == "thought" and cur:
                    _commit()

                if label == "thought":
                    cur["thought"] = _truncate(value, 600)
                elif label == "context":
                    cur["context"] = _truncate(value, 500)
                elif label == "action":
                    cur["action"] = _truncate(value, 200)
                elif label == "action input":
                    cur["action_input"] = _truncate(value, 1000)
                elif label == "observation":
                    cur["observation"] = _truncate(value, 500)
                elif label == "final answer":
                    cur["final_answer"] = _truncate(value, 1200)

            _commit()

        return all_steps

    def _looks_like_http(self, attrs: Dict[str, Any]) -> bool:
        for k in ("http.method", "http.url", "http.status_code", "http.route"):
            if k in attrs:
                return True
        return any(key.startswith("http.") for key in attrs.keys())

    def _extract_operational_metrics(
        self, span: ReadableSpan, attrs: Dict[str, Any]
    ) -> Dict[str, Any]:
        computed = {}

        duration_ns = span.end_time - span.start_time
        computed["neatlogs.metrics.duration_ms"] = duration_ns / 1_000_000

        # Skip event-based TTFT computation if already set live by a streaming patch
        # (OpenAI/Anthropic/LangChain/LiteLLM patches set these directly on the span)
        if attrs.get("neatlogs.llm.metrics.ttft_ms") is not None:
            return computed

        # Google GenAI emits "gen_ai.content.chunk" on every text chunk — use for TTFT + streaming_time_to_generate
        chunk_timestamps = []
        if span.events:
            for event in span.events:
                if event.name == "gen_ai.content.chunk":
                    chunk_timestamps.append(event.timestamp)

        if chunk_timestamps:
            first_ns = chunk_timestamps[0]
            ttft_ms = round((first_ns - span.start_time) / 1_000_000, 3)
            computed["neatlogs.llm.metrics.ttft_ms"] = ttft_ms

            if len(chunk_timestamps) >= 2:
                last_ns = chunk_timestamps[-1]
                stg_ms = round((last_ns - first_ns) / 1_000_000, 3)
                computed["neatlogs.llm.metrics.streaming_time_to_generate_ms"] = stg_ms

        return computed

    def _upcycle_events(self, span: ReadableSpan) -> Dict[str, Any]:
        upcycled: Dict[str, Any] = {}
        retriever_docs: List[Dict[str, Any]] = []

        for event in span.events:
            if event.name == "db.query.result":
                e_attrs = event.attributes
                doc: Dict[str, Any] = {
                    "timestamp": (
                        event.timestamp.isoformat()
                        if hasattr(event.timestamp, "isoformat")
                        else str(event.timestamp)
                    )
                }
                if "db.query.result.id" in e_attrs:
                    doc["id"] = e_attrs["db.query.result.id"]
                if "db.query.result.distance" in e_attrs:
                    doc["distance"] = e_attrs["db.query.result.distance"]
                if "db.query.result.document" in e_attrs:
                    doc["document"] = e_attrs["db.query.result.document"]
                if "db.query.result.metadata" in e_attrs:
                    metadata = e_attrs["db.query.result.metadata"]
                    try:
                        doc["metadata"] = (
                            json.loads(metadata) if isinstance(metadata, str) else metadata
                        )
                    except Exception:
                        doc["metadata"] = str(metadata)

                for field in ["_id", "title", "text", "category", "_score"]:
                    if field in e_attrs:
                        doc[field] = e_attrs[field]

                retriever_docs.append(doc)

            elif event.name == "db.search.result":
                e_attrs = event.attributes
                doc = {
                    "timestamp": (
                        event.timestamp.isoformat()
                        if hasattr(event.timestamp, "isoformat")
                        else str(event.timestamp)
                    )
                }
                if "db.search.query.id" in e_attrs:
                    doc["query_id"] = e_attrs["db.search.query.id"]
                if "db.search.result.id" in e_attrs:
                    doc["result_id"] = e_attrs["db.search.result.id"]
                if "db.search.result.distance" in e_attrs:
                    doc["distance"] = e_attrs["db.search.result.distance"]
                if "db.search.result.entity" in e_attrs:
                    doc["entity"] = e_attrs["db.search.result.entity"]
                retriever_docs.append(doc)

            elif event.name == "db.query.embeddings":
                e_attrs = event.attributes
                vector = e_attrs.get("db.query.embeddings.vector") or e_attrs.get("vector")
                if vector and isinstance(vector, (list, tuple, bytes)):
                    upcycled["neatlogs.db.query.embeddings.dimension"] = len(vector)
                    if self.debug:
                        self.logger.debug(f"Calculated embedding dimension: {len(vector)}")

        if retriever_docs:
            upcycled["retrieval_documents"] = json.dumps(retriever_docs)

        return upcycled

    def _apply_namespace_mapping(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        unified: Dict[str, Any] = {k: v for k, v in attrs.items() if k.startswith("neatlogs.")}
        mappings = self.mapping.get("mappings", {})

        consumed: set[str] = set()
        self._map_recursive(mappings, attrs, unified, consumed)

        keep_as_is = self.mapping.get("keep_as_is", {}).get("attributes", [])
        for key in keep_as_is:
            if key in attrs:
                unified[key] = attrs[key]

        ignore_patterns = self.mapping.get("ignore", {}).get("patterns", [])
        for key, value in attrs.items():
            if key.startswith("neatlogs.") or key in unified or key in keep_as_is:
                continue
            if key in consumed:
                continue

            should_ignore = False
            for pattern in ignore_patterns:
                if pattern.endswith("*") and key.startswith(pattern[:-1]):
                    should_ignore = True
                    break
                if key == pattern:
                    should_ignore = True
                    break

            # Unmapped attributes are not copied — they stay on the OTel span
            # natively and are exported via OTLP as-is.

        # Ensure neatlogs.span.kind is always set (for simplified view)
        if "neatlogs.span.kind" not in unified:
            oi_kind = attrs.get("openinference.span.kind")
            if oi_kind:
                unified["neatlogs.span.kind"] = str(oi_kind).lower()
            else:
                # Fallback: infer from span name
                from ..span_kinds.mapping import infer_span_kind_from_name
                span_name = attrs.get("_span_name", "")  # Will be set by span processor
                if span_name:
                    inferred_kind = infer_span_kind_from_name(span_name)
                    unified["neatlogs.span.kind"] = inferred_kind.lower()
        
        # Detect RERANKER operations from llm.request.type or gen_ai.operation.name
        llm_request_type = attrs.get("llm.request.type", "").lower()
        gen_ai_operation = attrs.get("gen_ai.operation.name", "").lower()
        span_name_lower = attrs.get("_span_name", "").lower()
        
        if llm_request_type == "rerank" or gen_ai_operation == "rerank" or "rerank" in span_name_lower:
            unified["neatlogs.span.kind"] = "reranker"
        
        span_kind = (
            attrs.get("neatlogs.span.kind") or attrs.get("openinference.span.kind") or ""
        ).lower()
        if span_kind not in ("embedding", "retriever", "vector_store"):
            unified.pop("neatlogs.vectordb.embedding_model", None)

        if self.debug:
            self.logger.debug(
                "[ScopeDetectionFinal] span_name=%s scope=%s framework=%s",
                attrs.get("_span_name"),
                attrs.get("neatlogs.instrumentation.name"),
                unified.get("neatlogs.framework"),
            )

        self._fill_provider_gaps(attrs, unified)

        return unified

    def _fill_provider_gaps(self, attrs: Dict[str, Any], unified: Dict[str, Any]) -> None:
        """
        Fill neatlogs.llm.provider and neatlogs.llm.system when OpenInference doesn't set them.

        OpenInference sets llm.provider for openai/anthropic/google_genai/bedrock/cohere but
        NOT for groq, mistralai, vertexai. The scope-based detection in
        enrich_with_scope_detection() already writes neatlogs.provider from the instrumentation
        scope name, so we use that as a fallback. Model-name inference is the last resort.
        """
        # --- neatlogs.llm.provider ---
        if not unified.get("neatlogs.llm.provider"):
            # Fallback 1: scope-detected provider (set by enrich_with_scope_detection)
            scope_provider = unified.get("neatlogs.provider") or attrs.get("neatlogs.provider", "")
            if scope_provider:
                unified["neatlogs.llm.provider"] = scope_provider
            else:
                # Fallback 2: infer from model name prefix
                model = (
                    attrs.get("llm.model_name")
                    or attrs.get("gen_ai.request.model")
                    or attrs.get("llm.model")
                    or ""
                )
                inferred = self._infer_provider_from_model(str(model))
                if inferred:
                    unified["neatlogs.llm.provider"] = inferred

        # --- neatlogs.llm.system ---
        if not unified.get("neatlogs.llm.system"):
            provider = (
                unified.get("neatlogs.llm.provider")
                or unified.get("neatlogs.provider")
                or ""
            ).lower()
            system = _PROVIDER_TO_SYSTEM.get(provider, "")
            if system:
                unified["neatlogs.llm.system"] = system

    def _infer_provider_from_model(self, model: str) -> str:
        """Infer LLM provider from model name prefix as a last resort."""
        if not model:
            return ""
        m = model.lower()
        # OpenAI model families
        if m.startswith(("gpt-", "o1-", "o3-", "o4-", "text-embedding-", "text-davinci-")):
            return "openai"
        # Anthropic
        if m.startswith("claude-"):
            return "anthropic"
        # Google
        if m.startswith(("gemini-", "gemma-")):
            return "google"
        # Mistral
        if m.startswith(("mistral-", "mixtral-")):
            return "mistralai"
        # Cohere
        if m.startswith(("command-", "embed-english", "embed-multilingual")):
            return "cohere"
        # Bedrock model IDs (e.g. "anthropic.claude-3-5-sonnet-v1:0", "meta.llama3-8b-instruct-v1:0")
        if m.startswith(("anthropic.", "meta.", "amazon.", "nova-", "titan-")):
            return "aws"
        # xAI
        if m.startswith("grok-"):
            return "xai"
        # DeepSeek
        if m.startswith("deepseek-"):
            return "deepseek"
        return ""

    def _map_recursive(
        self,
        mapping_tier: Dict[str, Any],
        source: Dict[str, Any],
        target: Dict[str, Any],
        consumed: set[str],
    ):
        for _, config in mapping_tier.items():
            if not isinstance(config, dict):
                continue

            if (
                isinstance(config, dict)
                and "mappings" in config
                and isinstance(config["mappings"], dict)
            ):
                self._map_recursive(config["mappings"], source, target, consumed)

            if isinstance(config, dict):
                for child_key, child_cfg in config.items():
                    if child_key in (
                        "mappings",
                        "description",
                        "priority",
                        "values",
                        "sources",
                        "target",
                        "indexed",
                        "template",
                        "target_content",
                        "target_template",
                    ):
                        continue
                    if not isinstance(child_cfg, dict):
                        continue
                    if any(k in child_cfg for k in ("target", "sources", "mappings", "indexed")):
                        self._map_recursive({child_key: child_cfg}, source, target, consumed)
                        continue

                    # Some tiers are purely organizational (e.g. metrics.llm.time_per_output_token)
                    # and don't contain leaf mapping keys until 2+ levels down. Always descend into
                    # nested dicts here to reach those leaf mappings.
                    nested_tier = {k: v for k, v in child_cfg.items() if isinstance(v, dict)}
                    if nested_tier:
                        self._map_recursive(nested_tier, source, target, consumed)

            target_key = config.get("target")
            if not target_key:
                continue

            if isinstance(target_key, str) and "{span_kind}" in target_key:
                resolved_kind = target.get("neatlogs.span.kind")
                if not resolved_kind:
                    resolved_kind = source.get("openinference.span.kind")
                    if isinstance(resolved_kind, str):
                        resolved_kind = resolved_kind.strip()
                if not resolved_kind:
                    continue
                target_key = target_key.replace("{span_kind}", str(resolved_kind))

            if config.get("indexed"):
                self._process_indexed_mapping(config, source, target, consumed)
                continue

            sources = config.get("sources", [])
            if isinstance(sources, list):
                for src_key in sources:
                    if src_key in source:
                        # Skip output.value if _skip_output_value flag is set (for real embedding operations)
                        if src_key == "output.value":
                            skip_flag = source.get("_skip_output_value")
                            
                            if skip_flag:
                                if self.debug:
                                    span_name = source.get("_span_name", "unknown")
                                    self.logger.debug(f"[AttributeProcessor] Skipping output.value for '{span_name}' due to _skip_output_value flag")
                                consumed.add(src_key)  # Mark as consumed but don't copy value
                                break
                        
                        val = source[src_key]
                        if "values" in config:
                            val = config["values"].get(val, val)
                        if src_key in ("input.value", "output.value"):
                            val = self._sanitize_io_value(val)
                        target[target_key] = val
                        consumed.add(src_key)
                        break

    def _process_indexed_mapping(
        self,
        config: Dict[str, Any],
        source: Dict[str, Any],
        target: Dict[str, Any],
        consumed: set[str],
    ):
        target_template = config.get("target")
        target_content_template = config.get("target_content")
        sources_config = config.get("sources")

        for i in range(20):
            found_any = False

            if isinstance(sources_config, list):
                if target_content_template:
                    role_val = None
                    content_val = None

                    for src_template in sources_config:
                        src_key = src_template.replace("{i}", str(i))
                        if src_key not in source:
                            continue
                        val = source[src_key]

                        if role_val is None and (
                            src_key.endswith(".role") or src_key.endswith(".message.role")
                        ):
                            role_val = val
                            consumed.add(src_key)
                        elif content_val is None and (
                            src_key.endswith(".content") or src_key.endswith(".message.content")
                        ):
                            content_val = val
                            consumed.add(src_key)

                        if role_val is not None and content_val is not None:
                            break

                    if role_val is not None:
                        target[target_template.replace("{i}", str(i))] = role_val
                        found_any = True
                    if content_val is not None:
                        target[target_content_template.replace("{i}", str(i))] = content_val
                        found_any = True
                else:
                    for src_template in sources_config:
                        src_key = src_template.replace("{i}", str(i))
                        if src_key in source:
                            target_key = target_template.replace("{i}", str(i))
                            target[target_key] = source[src_key]
                            consumed.add(src_key)
                            found_any = True
                            break

            elif isinstance(sources_config, dict):
                item_data: Dict[str, Any] = {}
                for sub_key, sub_sources in sources_config.items():
                    for src_template in sub_sources:
                        src_key = src_template.replace("{i}", str(i))
                        if src_key in source:
                            item_data[sub_key] = source[src_key]
                            consumed.add(src_key)
                            found_any = True
                            break

                if item_data:
                    for sub_key, sub_val in item_data.items():
                        target_key = f"{target_template.replace('{i}', str(i))}.{sub_key}"
                        target[target_key] = sub_val

            if not found_any:
                break
    
    def _filter_embedding_vectors(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out massive embedding vectors that cause memory/network issues.
        
        Removes:
        - embedding.embeddings.*.embedding.vector (4KB-4MB per embedding)
        - Any array with >1000 elements (likely a vector)
        """
        filtered = {}
        for key, value in attrs.items():
            # Skip embedding vector keys
            if ".embedding.vector" in key or ".embeddings." in key:
                if self.debug:
                    self.logger.debug(f"[FILTER] Dropped embedding vector key: {key}")
                continue
            
            # Skip large arrays (likely embedding vectors)
            if isinstance(value, (list, tuple)) and len(value) > 1000:
                if self.debug:
                    self.logger.debug(f"[FILTER] Dropped large array ({len(value)} elements): {key}")
                continue
            
            filtered[key] = value
        
        return filtered
