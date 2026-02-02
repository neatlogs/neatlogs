import json
import logging
import re
from typing import Dict, Any, List, Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind
from opentelemetry import metrics


class UnifiedAttributeProcessor:

    def __init__(
        self,
        mapping_config: Dict[str, Any],
        pricing_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        self.mapping = mapping_config
        self.pricing = pricing_config or {}
        self.debug = debug
        self.logger = logging.getLogger(__name__)

        self.meter = metrics.get_meter("neatlogs.sdk")

        self.time_per_token_histogram = self.meter.create_histogram(
            name="gen_ai.server.time_per_output_token",
            unit="ms",
            description="Inter-token latency for streaming LLM responses (SDK-computed)",
        )

    def process(self, span: ReadableSpan) -> Dict[str, Any]:
        res_attrs = dict(span.resource.attributes) if span.resource else {}
        attrs = {**res_attrs, **dict(span.attributes)}

        attrs = self._normalize_conventions(span, attrs)

        computed_metrics = self._extract_operational_metrics(span, attrs)
        attrs.update(computed_metrics)

        event_attrs = self._upcycle_events(span)
        attrs.update(event_attrs)

        attrs = self._apply_cost_fallback(attrs)

        try:
            from ..config import enrich_invocation_parameters
            enrich_invocation_parameters(attrs, enable_enrichment=True)
        except Exception as e:
            self.logger.warning(f"Failed to enrich invocation parameters: {e}")

        unified = self._apply_namespace_mapping(attrs)
        self._add_intermediate_steps(unified)
        return unified

    def _normalize_conventions(self, span: ReadableSpan, attrs: Dict[str, Any]) -> Dict[str, Any]:
        if span.kind == SpanKind.CLIENT and self._looks_like_http(attrs):
            attrs["openinference.span.kind"] = "HTTP"

        if "openinference.span.kind" not in attrs and any(
            k.startswith("crewai.crew.") for k in attrs.keys()
        ):
            attrs["openinference.span.kind"] = "CHAIN"

        self._add_crewai_token_usage_fallback(attrs)
        tool_calls: Dict[int, Dict[str, Any]] = {}
        oi_tool_re = re.compile(
            r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.function\.(name|arguments)$"
        )
        ol_tool_re = re.compile(r"^gen_ai\.completion\.(\d+)\.tool_calls\.(\d+)\.(id|name|arguments)$")

        keys_to_remove: List[str] = []
        for k, v in list(attrs.items()):
            m = oi_tool_re.match(k)
            if m:
                _msg_idx, call_idx, field = m.groups()
                idx = int(call_idx)
                tool_calls.setdefault(idx, {})[field] = v
                keys_to_remove.append(k)
                continue

            m = ol_tool_re.match(k)
            if m:
                _comp_idx, call_idx, field = m.groups()
                idx = int(call_idx)
                tool_calls.setdefault(idx, {})[field] = v
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
                    td.setdefault("input_schema", schema.get("input_schema") or schema.get("parameters"))
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

        if "openinference.span.kind" not in attrs and "traceloop.span.kind" in attrs:
            attrs["openinference.span.kind"] = attrs["traceloop.span.kind"]

        if "openinference.span.kind" not in attrs:
            db_system = attrs.get("db.system")
            if isinstance(db_system, str) and db_system.lower() in {
                "chroma",
                "chromadb",
                "pinecone",
                "qdrant",
                "milvus",
                "marqo",
                "weaviate",
                "pgvector",
                "elasticsearch",
            }:
                attrs["openinference.span.kind"] = "RETRIEVER"

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
                        if "arguments" in entity_input and isinstance(entity_input["arguments"], dict):
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

        if has_mcp_signal and "traceloop.entity.output" in attrs and "mcp.response.value" not in attrs:
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

            attrs["_skip_output_value"] = True

        if db_system == "chroma":
            doc_attrs = {}
            for key in [
                "db.chroma.add.ids_count",
                "db.chroma.add.embeddings_count",
                "db.chroma.add.metadatas_count",
                "db.chroma.add.documents_count",
                "db.chroma.query.n_results",
            ]:
                if key in attrs:
                    doc_attrs[key.split(".")[-1]] = attrs[key]

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

    def _add_crewai_token_usage_fallback(self, attrs: Dict[str, Any]) -> None:
        """
        Parse CrewAI aggregate token usage strings like:
          "total_tokens=67305 prompt_tokens=46983 cached_prompt_tokens=0 completion_tokens=20322 successful_requests=27"
        and map them to OpenInference-style token keys:
          - llm.token_count.prompt / completion / total
          - llm.token_count.prompt_details.cache_read (best-effort)
        Only applies when token_count fields are not already present.
        """
        usage = attrs.get("crewai.crew.token_usage")
        if not isinstance(usage, str) or not usage:
            return

        if any(k in attrs for k in ("llm.token_count.prompt", "llm.token_count.completion", "llm.token_count.total")):
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

    def _collect_role_texts(
        self, unified: Dict[str, Any], prefix: str, role: str
    ) -> List[str]:
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

    def _extract_operational_metrics(self, span: ReadableSpan, attrs: Dict[str, Any]) -> Dict[str, Any]:
        computed = {}

        duration_ns = span.end_time - span.start_time
        computed["neatlogs.metrics.duration_ms"] = duration_ns / 1_000_000

        chunk_timestamps = []
        if span.events:
            for event in span.events:
                if event.name == "llm.content.completion.chunk":
                    chunk_timestamps.append(event.timestamp / 1_000_000)

        if len(chunk_timestamps) > 1:
            diffs = [chunk_timestamps[i] - chunk_timestamps[i - 1] for i in range(1, len(chunk_timestamps))]
            mean_gap_ms = sum(diffs) / len(diffs)
            rounded_value = round(mean_gap_ms, 3)
            computed["gen_ai.server.time_per_output_token"] = rounded_value

            metric_attributes = {
                "trace_id": f"{span.context.trace_id:032x}",
                "span_id": f"{span.context.span_id:016x}",
            }
            model = attrs.get("gen_ai.request.model") or attrs.get("llm.model_name")
            if model:
                metric_attributes["llm_model"] = model

            self.time_per_token_histogram.record(rounded_value, attributes=metric_attributes)

        return computed

    def _upcycle_events(self, span: ReadableSpan) -> Dict[str, Any]:
        upcycled: Dict[str, Any] = {}
        retriever_docs: List[Dict[str, Any]] = []

        for event in span.events:
            if event.name == "db.query.result":
                e_attrs = event.attributes
                doc: Dict[str, Any] = {
                    "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp)
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
                        doc["metadata"] = json.loads(metadata) if isinstance(metadata, str) else metadata
                    except Exception:
                        doc["metadata"] = str(metadata)

                for field in ["_id", "title", "text", "category", "_score"]:
                    if field in e_attrs:
                        doc[field] = e_attrs[field]

                retriever_docs.append(doc)

            elif event.name == "db.search.result":
                e_attrs = event.attributes
                doc = {
                    "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp)
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

    def _apply_cost_fallback(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        if "llm.cost.total" in attrs:
            return attrs

        model = attrs.get("llm.model_name")
        prompt_tokens = attrs.get("llm.token_count.prompt")
        completion_tokens = attrs.get("llm.token_count.completion")

        if not model or not self.pricing:
            return attrs

        chat_pricing = self.pricing.get("chat", {})
        model_lower = model.lower()
        prices = None

        if model in chat_pricing:
            prices = chat_pricing[model]
        else:
            for model_key, p in chat_pricing.items():
                if model_key.lower() in model_lower or model_lower.startswith(model_key.lower()):
                    prices = p
                    break

        if prices and (prompt_tokens is not None or completion_tokens is not None):
            p_tokens = prompt_tokens or 0
            c_tokens = completion_tokens or 0
            prompt_cost = (p_tokens / 1000) * prices.get("promptPrice", 0)
            completion_cost = (c_tokens / 1000) * prices.get("completionPrice", 0)
            attrs["llm.cost.prompt"] = round(prompt_cost, 6)
            attrs["llm.cost.completion"] = round(completion_cost, 6)
            attrs["llm.cost.total"] = round(prompt_cost + completion_cost, 6)

        return attrs

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

            if not should_ignore:
                unified[f"neatlogs.raw.{key}"] = value

        span_kind = (attrs.get("neatlogs.span.kind") or attrs.get("openinference.span.kind") or "").lower()
        if span_kind not in ("embedding", "retriever"):
            unified.pop("neatlogs.vectordb.embedding_model", None)

        KNOWN_FRAMEWORKS = {
            'langchain',
            'llamaindex',
            'crewai',
            'haystack',
            'agno',
            'openai-agents'
        }

        gen_ai_system = attrs.get("gen_ai.system") or unified.get("neatlogs.llm.system") or ""
        if isinstance(gen_ai_system, str) and gen_ai_system:
            gen_ai_system_lower = gen_ai_system.lower()
            if gen_ai_system_lower in KNOWN_FRAMEWORKS:
                unified["neatlogs.framework"] = gen_ai_system_lower

        return unified

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

            if isinstance(config, dict) and "mappings" in config and isinstance(config["mappings"], dict):
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

                    if any(
                        isinstance(grand_cfg, dict) and any(k in grand_cfg for k in ("target", "sources", "mappings", "indexed"))
                        for grand_cfg in child_cfg.values()
                    ):
                        nested_tier = {k: v for k, v in child_cfg.items() if isinstance(v, dict)}
                        self._map_recursive(nested_tier, source, target, consumed)

            target_key = config.get("target")
            if not target_key:
                continue

            if isinstance(target_key, str) and "{span_kind}" in target_key:
                resolved_kind = target.get("neatlogs.span.kind")
                if not resolved_kind:
                    resolved_kind = source.get("openinference.span.kind") or source.get("traceloop.span.kind")
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
                        val = source[src_key]
                        if "values" in config:
                            val = config["values"].get(val, val)
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

                        if role_val is None and (src_key.endswith(".role") or src_key.endswith(".message.role")):
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
