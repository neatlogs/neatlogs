"""
Neatlogs span processor.
"""

import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from .attribute_processor import UnifiedAttributeProcessor
from .exporter import NeatlogsExporter
from .logger import get_logger

logger = get_logger()


class NeatlogsSpanProcessor(SpanProcessor):
    def __init__(
        self,
        exporter: NeatlogsExporter,
        sample_rate: float = 1.0,
        debug: bool = False,
    ):
        self.exporter = exporter
        self.sample_rate = sample_rate
        self.debug = debug

        self._init_processor()

        self._log_raw_spans_enabled = self.debug or (
            os.getenv("NEATLOGS_LOG_RAW_SPANS", "").lower() in ["true", "1", "yes"]
        )
        self._raw_log_file_path = None
        self._raw_log_file_handle = None
        if self._log_raw_spans_enabled:
            self._raw_log_file_path = os.path.join(
                os.getcwd(), os.getenv("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_optimized.log")
            )
            try:
                self._raw_log_file_handle = open(self._raw_log_file_path, "a", encoding="utf-8")
            except Exception:
                self._raw_log_file_handle = None

        self._pending: List[Dict[str, Any]] = []
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._stop_background = threading.Event()
        self._dedupe_interval = float(os.getenv("NEATLOGS_DEDUPE_INTERVAL_S", "1.0"))
        self._dedupe_latency_ns = int(os.getenv("NEATLOGS_DEDUPE_LATENCY_MS", "2000")) * 1_000_000
        self._max_pending_spans = int(os.getenv("NEATLOGS_MAX_PENDING_SPANS", "5000"))
        self._pending_high_watermark = 0
        self._pending_dropped = 0
        self._background_thread = threading.Thread(target=self._background_flush_loop, daemon=True)
        self._background_thread.start()

        self.perf_stats = {
            "on_start_time": 0.0,
            "on_end_time": 0.0,
            "spans_processed": 0,
            "spans_exported": 0,
        }

    def _init_processor(self) -> None:
        base_path = os.path.dirname(os.path.dirname(__file__))

        mapping_path = os.path.join(base_path, "config", "attribute-mapping.json")
        try:
            with open(mapping_path, "r") as f:
                mapping_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load attribute-mapping.json: {e}")
            mapping_config = {}

        pricing_path = os.path.join(base_path, "config", "pricing.json")
        try:
            with open(pricing_path, "r") as f:
                pricing_config = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load pricing.json: {e}")
            pricing_config = {}

        self.unified_processor = UnifiedAttributeProcessor(
            mapping_config=mapping_config,
            pricing_config=pricing_config,
            debug=self.debug,
        )

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        start_time = time.perf_counter()
        try:
            span_kind = span.attributes.get("openinference.span.kind") if span.attributes else None
            is_llm_span = (
                span_kind == "LLM"
                or "chat" in span.name.lower()
                or "completion" in span.name.lower()
                or "generate" in span.name.lower()
                or "embedding" in span.name.lower()
            )
            if not is_llm_span:
                return

            from opentelemetry.context import get_current, get_value

            from ..prompt.template import PromptContext, UserPromptContext

            ctx = get_current()
            variables_json = get_value("neatlogs.prompt_variables", context=ctx)
            template = get_value("neatlogs.prompt_template", context=ctx)
            version_val = get_value("neatlogs.prompt_version", context=ctx)

            if not variables_json:
                captured_vars = PromptContext.get_variables()
                if captured_vars:
                    variables_json = json.dumps(captured_vars, default=str)

            if not template:
                captured_template = PromptContext.get_template()
                if captured_template:
                    template = captured_template

            # Capture user prompt template and variables (separate from system prompt)
            user_template = get_value("neatlogs.user_prompt_template", context=ctx)
            user_variables_json = get_value("neatlogs.user_prompt_variables", context=ctx)

            if not user_template:
                captured_user_template = UserPromptContext.get_template()
                if captured_user_template:
                    user_template = captured_user_template

            if not user_variables_json:
                captured_user_vars = UserPromptContext.get_variables()
                if captured_user_vars:
                    user_variables_json = json.dumps(captured_user_vars, default=str)

            if self.debug:
                logger.debug(f"[SpanProcessor.on_start] LLM span '{span.name}' starting")
                logger.debug(f"  variables_json from context: {variables_json}")
                logger.debug(f"  template from context: {template}")
                logger.debug(f"  version from context: {version_val}")
                logger.debug(f"  user_template from context: {user_template}")
                logger.debug(f"  user_variables_json from context: {user_variables_json}")

            if variables_json:
                span.set_attribute("llm.prompt_template_variables", variables_json)
            if template:
                span.set_attribute("llm.prompt_template", template)
            if version_val:
                span.set_attribute("llm.prompt_template.version", version_val)
            if user_template:
                span.set_attribute("llm.user_prompt_template", user_template)
            if user_variables_json:
                span.set_attribute("llm.user_prompt_template_variables", user_variables_json)
        finally:
            self.perf_stats["on_start_time"] += time.perf_counter() - start_time

    def on_end(self, span: ReadableSpan) -> None:
        start_time = time.perf_counter()
        self.perf_stats["spans_processed"] += 1

        try:
            if self.debug:
                logger.debug(f"[SpanProcessor.on_end] Span ending: {span.name}")

            if self._raw_log_file_handle:
                try:
                    self._raw_log_file_handle.write(span.to_json() + "\n")
                    self._raw_log_file_handle.flush()
                except Exception as e:
                    logger.warning(f"Failed to write span to raw log file: {e}")

            if self.sample_rate < 1.0 and random.random() > self.sample_rate:
                return

            unified_attrs = self.unified_processor.process(span)

            # CRITICAL: Filter out large tokenized arrays for EMBEDDING/VECTOR_STORE spans
            # These arrays can be 8+ MB and should NOT be sent to Kafka
            nl_kind = unified_attrs.get("neatlogs.span.kind")
            if nl_kind in ("embedding", "vector_store"):
                # Check if this is a REAL embedding operation (flag set) or user's @span(kind="EMBEDDING")
                skip_output = unified_attrs.get("neatlogs.raw._skip_output_value") == True
                
                keys_to_remove = []
                for key in unified_attrs.keys():
                    # Remove massive tokenized input/output arrays
                    if (
                        "input_messages" in key or
                        "output_messages" in key or
                        "gen_ai.prompt" in key or
                        "gen_ai.completion" in key or
                        ".content" in key
                    ):
                        keys_to_remove.append(key)
                    # Only remove embedding input/output if it's a REAL embedding operation
                    elif skip_output and (
                        key == "neatlogs.embedding.input" or
                        key == "neatlogs.embedding.output" or
                        key == "neatlogs.raw.embedding.input" or
                        key == "neatlogs.raw.embedding.output"
                    ):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    unified_attrs.pop(key, None)
                
                if self.debug and keys_to_remove:
                    logger.debug(
                        f"[EMBEDDING Filter] Removed {len(keys_to_remove)} large attribute keys "
                        f"from {nl_kind} span (skip_output={skip_output}) to prevent 8+ MB payloads"
                    )

            nl_kind = unified_attrs.get("neatlogs.span.kind")
            if nl_kind not in ("llm", "embedding") and span.name != "PromptTemplate":
                for k in (
                    "neatlogs.llm.prompt_template",
                    "neatlogs.llm.prompt_template_variables",
                    "neatlogs.llm.prompt_template.version",
                    "neatlogs.raw.llm.prompt_template",
                    "neatlogs.raw.llm.prompt_template_variables",
                    "neatlogs.raw.llm.prompt_template.version",
                ):
                    unified_attrs.pop(k, None)
            else:
                for k in (
                    "neatlogs.raw.llm.token_count.prompt",
                    "neatlogs.raw.llm.token_count.completion",
                    "neatlogs.raw.llm.token_count.total",
                ):
                    unified_attrs.pop(k, None)

                for k in (
                    "neatlogs.raw.gen_ai.usage.input_tokens",
                    "neatlogs.raw.gen_ai.usage.output_tokens",
                ):
                    unified_attrs.pop(k, None)

            if span.name == "PromptTemplate":
                unified_attrs.setdefault("neatlogs.internal", True)
                unified_attrs["neatlogs.span.kind"] = "Neatlogs.INTERNAL"
                unified_attrs["neatlogs.span.kind"] = "Neatlogs.INTERNAL"

            # Include resource attributes for tags, session_id, user_id, etc.
            resource_attrs = {}
            if span.resource and span.resource.attributes:
                # Properly serialize OTel resource attributes (handles AttributeValue types)
                for key, value in span.resource.attributes.items():
                    # Convert AttributeValue to native Python types
                    if isinstance(value, (str, int, float, bool)):
                        resource_attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        resource_attrs[key] = list(value)
                    else:
                        resource_attrs[key] = str(value)
                
                # Debug logging for tags
                if self.debug and "neatlogs.tags" in resource_attrs:
                    logger.debug(f"[Tags] Span {span.name}: resource.neatlogs.tags = {resource_attrs['neatlogs.tags']}")
            
            trace_id = f"{span.context.trace_id:032x}"
            span_id = f"{span.context.span_id:016x}"
            parent_span_id = f"{span.parent.span_id:016x}" if span.parent else None

            # Guard against corrupted/self-referential parenting (seen in some LangChain/LangGraph
            # instrumentation edge cases where the SDK ends up exporting a span whose parent span id
            # equals its own span id). This creates cycles and breaks trace trees downstream.
            if parent_span_id == span_id:
                if self.debug:
                    logger.warning(
                        "[SpanProcessor] Detected self-parenting span. "
                        f"trace_id={trace_id} span_id={span_id} name={span.name}. "
                        "Setting parent_span_id=None."
                    )
                parent_span_id = None

            span_data = {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": span.name,
                "kind": (unified_attrs.get("neatlogs.span.kind", "UNKNOWN") or "UNKNOWN"),
                "start_time": span.start_time,
                "end_time": span.end_time,
                "duration_ns": span.end_time - span.start_time if span.end_time else None,
                "attributes": unified_attrs,
                "resource": {"attributes": resource_attrs},  # Add resource attributes
                "status": {
                    "code": span.status.status_code.name,
                    "description": span.status.description,
                },
                "events": (
                    [
                        {
                            "name": event.name,
                            "timestamp": event.timestamp,
                            "attributes": dict(event.attributes) if event.attributes else {},
                        }
                        for event in span.events
                        if not (
                            any(
                                e.name
                                in {
                                    "llm.content.completion.chunk",
                                    "gen_ai.content.completion.chunk",
                                    "neatlogs.gen_ai.content.completion.chunk",
                                }
                                for e in span.events
                            )
                            and event.name == "First Token Stream Event"
                        )
                    ]
                    if span.events
                    else []
                ),
            }

            with self._pending_lock:
                if len(self._pending) >= self._max_pending_spans:
                    dropped_span = self._pending.pop(0)
                    self._pending_dropped += 1
                    logger.warning(
                        f"Pending buffer full ({self._max_pending_spans} spans), "
                        f"dropping oldest span: {dropped_span.get('name')} "
                        f"(total dropped: {self._pending_dropped})"
                    )

                self._pending.append(span_data)
                curr = len(self._pending)
                if curr > self._pending_high_watermark:
                    self._pending_high_watermark = curr

                if not span.parent:
                    trace_id = span_data["trace_id"]
                    marker_span_id = f"completion_{trace_id[:16]}"

                    completion_marker = {
                        "trace_id": trace_id,
                        "span_id": marker_span_id,
                        "parent_span_id": None,
                        "name": "neatlogs.trace.complete",
                        "kind": "INTERNAL",
                        "start_time": span.end_time,
                        "end_time": span.end_time,
                        "duration_ns": 0,
                        "attributes": {
                            "neatlogs.trace.complete": True,
                            "neatlogs.internal": True,
                            "neatlogs.span.kind": "Neatlogs.INTERNAL",
                        },
                        "status": {"code": "OK", "description": ""},
                        "events": [],
                    }
                    self._pending.append(completion_marker)

                    if self.debug:
                        logger.debug(f"Added completion marker for trace {trace_id}")

            self._pending_event.set()
        finally:
            self.perf_stats["on_end_time"] += time.perf_counter() - start_time

    def _dedupe_and_rewrite(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_trace: Dict[str, List[Dict[str, Any]]] = {}
        for s in spans:
            by_trace.setdefault(s["trace_id"], []).append(s)

        out: List[Dict[str, Any]] = []
        for trace_id, trace_spans in by_trace.items():
            out.extend(self._dedupe_trace(trace_spans))

        out.sort(key=lambda s: (s["trace_id"], s.get("start_time") or 0, s["span_id"]))
        return out

    def _background_flush_loop(self) -> None:
        while not self._stop_background.is_set():
            self._pending_event.wait(timeout=self._dedupe_interval)
            self._pending_event.clear()
            self._flush_ready_spans()

    def _flush_ready_spans(self, force: bool = False) -> None:
        with self._pending_lock:
            pending = list(self._pending)
            self._pending.clear()

        if not pending:
            return

        now_ns = time.time_ns()
        cutoff_ns = now_ns - self._dedupe_latency_ns

        ready: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []

        for s in pending:
            end_ts = s.get("end_time")
            if force or (isinstance(end_ts, int) and end_ts <= cutoff_ns):
                ready.append(s)
            else:
                remaining.append(s)

        if not force:
            if len(remaining) > self._max_pending_spans:
                remaining.sort(key=lambda s: int(s.get("end_time") or 0))
                overflow = len(remaining) - self._max_pending_spans
                ready.extend(remaining[:overflow])
                remaining = remaining[overflow:]
            with self._pending_lock:
                self._pending.extend(remaining)
        else:
            ready.extend(remaining)

        if not ready:
            return

        rewritten = self._dedupe_and_rewrite(ready)
        for s in rewritten:
            self.exporter.export(s)
            self.perf_stats["spans_exported"] += 1

    def _dedupe_trace(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        spans_by_id = {s["span_id"]: s for s in spans}

        wrappers = [s for s in spans if self._is_openllmetry_provider_wrapper(s)]
        canonicals = [s for s in spans if self._is_openinference_canonical(s)]

        suppressed_wrapper_ids: set[str] = set()

        for w in wrappers:
            c = self._best_match_wrapper_to_canonical(w, canonicals)
            if not c:
                continue

            self._merge_wrapper_attrs_into_canonical(wrapper=w, canonical=c)
            suppressed_wrapper_ids.add(w["span_id"])

            # Re-parent any children of the wrapper to the wrapper's parent.
            replacement_parent = w.get("parent_span_id")
            for s in spans:
                if s.get("parent_span_id") == w["span_id"]:
                    s["parent_span_id"] = replacement_parent

        emitted = [s for s in spans if s["span_id"] not in suppressed_wrapper_ids]
        emitted = self._suppress_traceloop_entity_spans(emitted)
        emitted = self._suppress_overlapping_llm_spans(emitted)
        emitted = self._suppress_identical_llm_siblings(emitted)
        emitted = self._zero_duplicate_parent_tokens(emitted)
        emitted = self._normalize_framework_span_names(emitted)

        return emitted

    def _suppress_identical_llm_siblings(
        self, spans: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Suppress duplicate LLM sibling spans that share the same
        (parent_span_id, name, prompt_tokens, completion_tokens) and overlap in time.
        Keeps the span with the most children; re-parents the rest.
        Handles the cross-batch case where the provider canonical was flushed separately.
        """
        llm_spans = [
            s for s in spans if (s.get("attributes") or {}).get("neatlogs.span.kind") == "llm"
        ]
        if len(llm_spans) < 2:
            return spans

        children: Dict[str, List[Dict[str, Any]]] = {}
        for s in spans:
            pid = s.get("parent_span_id")
            if pid:
                children.setdefault(pid, []).append(s)

        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for s in llm_spans:
            attrs = s.get("attributes") or {}
            p = attrs.get("neatlogs.llm.token_count.prompt") or 0
            c = attrs.get("neatlogs.llm.token_count.completion") or 0
            if not p and not c:
                continue
            key = (s.get("parent_span_id"), s.get("name"), p, c)
            groups.setdefault(key, []).append(s)

        suppressed: set[str] = set()
        for key, group in groups.items():
            if len(group) < 2:
                continue
            has_overlap = any(
                self._score_time_overlap(group[i], group[j]) > 0
                for i in range(len(group))
                for j in range(i + 1, len(group))
            )
            if not has_overlap:
                continue
            keeper = max(group, key=lambda s: len(children.get(s["span_id"], [])))
            for s in group:
                if s["span_id"] == keeper["span_id"]:
                    continue
                suppressed.add(s["span_id"])
                for child in spans:
                    if child.get("parent_span_id") == s["span_id"]:
                        child["parent_span_id"] = keeper["span_id"]
                if self.debug:
                    logger.debug(
                        f"[Dedup] Identical sibling: suppressing {s['span_id']} "
                        f"({s.get('name')}) → keeping {keeper['span_id']}"
                    )

        return [s for s in spans if s["span_id"] not in suppressed]

    def _zero_duplicate_parent_tokens(
        self, spans: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Safety net: if an LLM span's direct parent is also LLM and has identical
        non-zero prompt AND completion tokens, zero out the parent's token and cost fields.
        The parent span is kept in the tree for structural integrity.
        """
        span_by_id = {s["span_id"]: s for s in spans}
        for s in spans:
            if (s.get("attributes") or {}).get("neatlogs.span.kind") != "llm":
                continue
            pid = s.get("parent_span_id")
            if not pid:
                continue
            parent = span_by_id.get(pid)
            if not parent:
                continue
            if (parent.get("attributes") or {}).get("neatlogs.span.kind") != "llm":
                continue
            sa = s.get("attributes") or {}
            pa = parent.get("attributes") or {}
            s_prompt = sa.get("neatlogs.llm.token_count.prompt") or 0
            s_comp = sa.get("neatlogs.llm.token_count.completion") or 0
            if not s_prompt and not s_comp:
                continue
            p_prompt = pa.get("neatlogs.llm.token_count.prompt") or 0
            p_comp = pa.get("neatlogs.llm.token_count.completion") or 0
            if s_prompt == p_prompt and s_comp == p_comp:
                for k in list(pa.keys()):
                    if k.startswith("neatlogs.llm.token_count") or k.startswith(
                        "neatlogs.llm.cost"
                    ):
                        pa[k] = 0
                if self.debug:
                    logger.debug(
                        f"[Dedup] Zeroed duplicate parent tokens: {parent['span_id']} "
                        f"({parent.get('name')}) same tokens as child {s['span_id']} "
                        f"({s.get('name')})"
                    )
        return spans

    def _normalize_framework_span_names(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reduce very long span names produced by some framework instrumentations.
        """
        for s in spans:
            name = s.get("name") or ""
            kind = s.get("kind") or (s.get("attributes") or {}).get("neatlogs.span.kind")
            if kind != "task" or not name.endswith(".task"):
                continue

            attrs = s.get("attributes") or {}
            if not any(k.startswith("neatlogs.raw.crewai.") for k in attrs.keys()):
                continue

            desc = name[: -len(".task")].rstrip()
            while desc.endswith("."):
                desc = desc[:-1].rstrip()

            if desc:
                attrs.setdefault("neatlogs.task.description", desc)
            s["name"] = "crewai.task"
            s["attributes"] = attrs

        return spans

    def _suppress_traceloop_entity_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id = {s["span_id"]: s for s in spans}

        entity_spans = [s for s in spans if self._is_traceloop_entity_span(s)]
        if not entity_spans:
            return spans

        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for s in spans:
            by_name.setdefault(s.get("name") or "", []).append(s)

        # IMPORTANT:
        # We used to *suppress* Traceloop entity spans (e.g. `.task`, `.tool`, `.workflow`) and merge
        # their attributes into the matching "base" span. This breaks parenting in practice because
        # traces are often exported in multiple chunks by end_time; the entity span (often long-lived)
        # arrives in a later chunk, but its children (shorter spans) have already been exported with
        # `parent_span_id` pointing at the entity span_id. Once suppressed, ClickHouse/UX sees those
        # children as "orphans" (missing parent in the trace).
        #
        # To preserve the trace tree, we keep the entity spans and ONLY merge their attributes into
        # the base span when we can match them.
        for e in entity_spans:
            base_name = self._traceloop_base_name(e.get("name") or "")
            if not base_name:
                continue
            candidates = by_name.get(base_name, [])
            if not candidates:
                continue

            c = self._best_match_entity_to_base(entity=e, bases=candidates)
            if not c:
                continue

            self._merge_traceloop_entity_attrs_into_base(entity=e, base=c)
            # NOTE: Do NOT re-parent children here either. Re-parenting is only safe when you have
            # the full trace buffered; otherwise it can create self-parenting (e.g. LangGraph.workflow
            # matched to LangGraph base span) or partial rewrites across chunks.

        return spans

    def _is_traceloop_entity_span(self, span_data: Dict[str, Any]) -> bool:
        name = span_data.get("name") or ""
        return name.endswith(".task") or name.endswith(".tool") or name.endswith(".workflow")

    def _traceloop_base_name(self, entity_name: str) -> Optional[str]:
        for suffix in (".task", ".tool", ".workflow"):
            if entity_name.endswith(suffix):
                return entity_name[: -len(suffix)]
        return None

    def _best_match_entity_to_base(
        self,
        entity: Dict[str, Any],
        bases: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        best = None
        best_score = 0
        for b in bases:
            score = self._score_time_overlap(entity, b)
            if entity.get("parent_span_id") and entity.get("parent_span_id") == b.get(
                "parent_span_id"
            ):
                score += 1
            if (
                self._start_delta_ns(entity, b) is not None
                and self._start_delta_ns(entity, b) <= 5_000_000
            ):
                score += 1
            if score > best_score:
                best_score = score
                best = b
        return best if best_score >= 3 else None

    def _score_time_overlap(self, a: Dict[str, Any], b: Dict[str, Any]) -> int:
        a_s, a_e = a.get("start_time"), a.get("end_time")
        b_s, b_e = b.get("start_time"), b.get("end_time")
        if not all(isinstance(x, int) for x in (a_s, a_e, b_s, b_e)):
            return 0
        latest_start = max(a_s, b_s)
        earliest_end = min(a_e, b_e)
        if earliest_end < latest_start:
            return 0
        overlap = earliest_end - latest_start
        dur_a = a_e - a_s
        dur_b = b_e - b_s
        shorter = min(dur_a, dur_b)
        if shorter <= 0:
            return 0
        return 3 if overlap >= (shorter // 2) else 1

    def _providers_match_fuzzy(self, a: str, b: str) -> bool:
        """Return True when two provider strings belong to the same family."""
        if a == b:
            return True
        # Containment (e.g. "azure_openai" contains "openai")
        if a in b or b in a:
            return True
        # Azure/OpenAI family: treat interchangeably
        _OPENAI_FAMILY = {"openai", "azure", "azure_openai"}
        if a in _OPENAI_FAMILY and b in _OPENAI_FAMILY:
            return True
        return False

    def _start_delta_ns(self, a: Dict[str, Any], b: Dict[str, Any]) -> Optional[int]:
        a_s, b_s = a.get("start_time"), b.get("start_time")
        if not (isinstance(a_s, int) and isinstance(b_s, int)):
            return None
        return abs(a_s - b_s)

    def _merge_traceloop_entity_attrs_into_base(
        self, entity: Dict[str, Any], base: Dict[str, Any]
    ) -> None:
        ea = entity.get("attributes", {})
        ba = base.get("attributes", {})
        for k, v in ea.items():
            if k == "neatlogs.span.kind":
                continue
            if k not in ba:
                ba[k] = v

        base["attributes"] = ba

    def _suppress_overlapping_llm_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        children: Dict[str, List[Dict[str, Any]]] = {}
        for s in spans:
            pid = s.get("parent_span_id")
            if pid:
                children.setdefault(pid, []).append(s)

        def has_http_child(sid: str) -> bool:
            for c in children.get(sid, []):
                a = c.get("attributes") or {}
                if a.get("neatlogs.span.kind") == "http":
                    return True
            return False

        llm_spans = [
            s for s in spans if (s.get("attributes") or {}).get("neatlogs.span.kind") == "llm"
        ]
        
        # Also handle LLM wrappers around EMBEDDING/RERANKER spans (e.g., openai.embeddings → CreateEmbeddings)
        suppressed_embedding_wrappers: set[str] = set()
        for llm_span in llm_spans:
            llm_children = children.get(llm_span["span_id"], [])
            for child in llm_children:
                child_kind = (child.get("attributes") or {}).get("neatlogs.span.kind", "").lower()
                if child_kind in ("embedding", "reranker"):
                    # This LLM span is a wrapper around an embedding/reranker - suppress it
                    suppressed_embedding_wrappers.add(llm_span["span_id"])
                    if self.debug:
                        logger.debug(
                            f"[Dedup] Suppressing LLM wrapper {llm_span['span_id']} "
                            f"(has {child_kind} child {child['span_id']})"
                        )
                    break
        
        if len(llm_spans) < 2 and not suppressed_embedding_wrappers:
            return spans

        provider_like = [s for s in llm_spans if has_http_child(s["span_id"])]
        framework_like = [s for s in llm_spans if not has_http_child(s["span_id"])]
        if not provider_like or not framework_like:
            return spans

        suppressed: set[str] = set()

        for fw in framework_like:
            best = None
            best_score = 0
            fwa = fw.get("attributes") or {}
            fw_provider = (
                fwa.get("neatlogs.llm.provider")
                or fwa.get("neatlogs.llm.system")
                or fwa.get("neatlogs.raw.gen_ai.system")
                or ""
            ).lower()

            for pv in provider_like:
                score = self._score_time_overlap(fw, pv)
                if fw.get("parent_span_id") and fw.get("parent_span_id") == pv.get(
                    "parent_span_id"
                ):
                    score += 2
                pva = pv.get("attributes") or {}
                pv_provider = (
                    pva.get("neatlogs.llm.provider")
                    or pva.get("neatlogs.llm.system")
                    or pva.get("neatlogs.raw.gen_ai.system")
                    or ""
                ).lower()
                if fw_provider and pv_provider and self._providers_match_fuzzy(
                    fw_provider, pv_provider
                ):
                    score += 1

                if score > best_score:
                    best_score = score
                    best = pv

            if not best or best_score < 4:
                continue

            self._merge_framework_llm_into_provider(framework=fw, provider=best)
            suppressed.add(fw["span_id"])

            # Re-parent children of the suppressed framework span.
            # If a child IS the best (provider) span, promote it to fw's parent instead
            # to avoid self-parenting.
            fw_parent = fw.get("parent_span_id")
            for s in spans:
                if s.get("parent_span_id") == fw["span_id"]:
                    if s["span_id"] == best["span_id"]:
                        s["parent_span_id"] = fw_parent
                    else:
                        s["parent_span_id"] = best["span_id"]

        # Combine both suppression sets
        all_suppressed = suppressed | suppressed_embedding_wrappers
        return [s for s in spans if s["span_id"] not in all_suppressed]

    def _merge_framework_llm_into_provider(
        self, framework: Dict[str, Any], provider: Dict[str, Any]
    ) -> None:
        fa = framework.get("attributes", {})
        pa = provider.get("attributes", {})

        prefer_oi_llm_names = os.getenv("NEATLOGS_PREFER_OPENINFERENCE_LLM_NAMES", "").lower() in (
            "true",
            "1",
            "yes",
        )

        def _as_num(v: Any) -> float:
            try:
                if v is None:
                    return 0.0
                if isinstance(v, bool):
                    return float(int(v))
                return float(v)
            except Exception:
                return 0.0

        def _prefer_if_missing_or_zero(key: str) -> None:
            fv = fa.get(key)
            if fv is None:
                return
            pv = pa.get(key)
            if pv is None or _as_num(pv) <= 0:
                pa[key] = fv

        # Prefer OpenInference token counts when the provider wrapper doesn't have them populated.
        for k in (
            "neatlogs.llm.token_count.prompt",
            "neatlogs.llm.token_count.completion",
            "neatlogs.llm.token_count.total",
            "neatlogs.llm.token_count.reasoning",
            "neatlogs.llm.token_count.cache_read",
            "neatlogs.llm.token_count.cached_read",
            "neatlogs.llm.token_count.cache_write",
            "neatlogs.llm.token_count.cached_write",
            "neatlogs.llm.token_count.audio",
            "neatlogs.llm.token_count.completion_audio",
        ):
            _prefer_if_missing_or_zero(k)

        for k, v in fa.items():
            if k == "neatlogs.span.kind":
                continue
            if k not in pa:
                pa[k] = v

        if prefer_oi_llm_names:
            fw_name = str(framework.get("name") or "").strip()
            pv_name = str(provider.get("name") or "").strip()

            fw_name_l = fw_name.lower()
            pv_name_l = pv_name.lower()

            # Common provider-side / generic names we prefer to replace with the OpenInference name.
            generic_provider_names = {"messages"}

            if fw_name and pv_name and fw_name_l != pv_name_l:
                if fw_name.startswith("Chat") and pv_name_l in generic_provider_names:
                    pa.setdefault("neatlogs.dedupe.original_span_name", pv_name)
                    pa.setdefault("neatlogs.dedupe.merged_from_span_id", framework.get("span_id"))
                    provider["name"] = fw_name

        provider["attributes"] = pa

    def _is_openllmetry_provider_wrapper(self, span_data: Dict[str, Any]) -> bool:
        attrs = span_data.get("attributes", {})
        name = (span_data.get("name") or "").lower()
        # Traceloop entity spans (e.g. LangGraph `.workflow` / Runnable `.task`) must NOT be treated
        # as OpenLLMetry provider wrappers. If we suppress them during partial/interval flushes,
        # their children may have already been exported with `parent_span_id` pointing to the entity
        # span, which produces "orphan" spans downstream (ClickHouse/UI). These are handled by
        # `_suppress_traceloop_entity_spans` instead (which preserves the tree).
        if self._is_traceloop_entity_span(span_data):
            return False
        if (
            "." in name
            and attrs.get("neatlogs.raw.openinference.span.kind") not in ("LLM", "EMBEDDING")
            and attrs.get("neatlogs.llm.request_type") is not None
        ):
            return True
        return False

    def _is_openinference_canonical(self, span_data: Dict[str, Any]) -> bool:
        attrs = span_data.get("attributes", {})
        raw_kind = attrs.get("neatlogs.raw.openinference.span.kind")
        if raw_kind in ("LLM", "EMBEDDING"):
            return True

        kind = attrs.get("neatlogs.span.kind")
        if kind == "embedding":
            return True

        if kind == "llm" and (attrs.get("neatlogs.llm.provider") is not None):
            return True

        return False

    def _best_match_wrapper_to_canonical(
        self,
        wrapper: Dict[str, Any],
        canonicals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        best = None
        best_score = 0

        for c in canonicals:
            score = self._score_pair(wrapper, c)
            if score > best_score:
                best_score = score
                best = c

        return best if best_score >= 6 else None

    def _score_pair(self, w: Dict[str, Any], c: Dict[str, Any]) -> int:
        wa = w.get("attributes", {})
        ca = c.get("attributes", {})

        score = 0

        if c.get("parent_span_id") == w.get("span_id"):
            score += 6
        if w.get("parent_span_id") == c.get("span_id"):
            score += 6

        w_provider = (
            wa.get("neatlogs.llm.system") or wa.get("neatlogs.llm.provider") or ""
        ).lower()
        c_provider = (
            ca.get("neatlogs.llm.system") or ca.get("neatlogs.llm.provider") or ""
        ).lower()
        if w_provider and c_provider and w_provider == c_provider:
            score += 2

        w_req_model = wa.get("neatlogs.raw.gen_ai.request.model") or wa.get(
            "neatlogs.llm.model_name"
        )
        c_resp_model = ca.get("neatlogs.raw.llm.model_name") or ca.get("neatlogs.llm.model_name")
        if w_req_model and c_resp_model:
            if str(c_resp_model).startswith(str(w_req_model)):
                score += 1

        w_s, w_e = w.get("start_time"), w.get("end_time")
        c_s, c_e = c.get("start_time"), c.get("end_time")
        if all(isinstance(x, int) for x in (w_s, w_e, c_s, c_e)):
            latest_start = max(w_s, c_s)
            earliest_end = min(w_e, c_e)
            if earliest_end >= latest_start:
                score += 1

        return score

    def _merge_wrapper_attrs_into_canonical(
        self, wrapper: Dict[str, Any], canonical: Dict[str, Any]
    ) -> None:
        wa = wrapper.get("attributes", {})
        ca = canonical.get("attributes", {})

        override_keys = {
            "neatlogs.llm.model_name",
            "neatlogs.framework",
        }

        for k, v in wa.items():
            if k in override_keys:
                ca[k] = v
                continue
            if k not in ca:
                ca[k] = v

        def _get(key: str):
            if key in ca and ca[key] is not None:
                return ca[key]
            return None

        def _set_if_missing(key: str, value):
            if key not in ca and value is not None:
                ca[key] = value

        _set_if_missing(
            "neatlogs.llm.token_count.prompt", _get("neatlogs.raw.gen_ai.usage.input_tokens")
        )
        _set_if_missing(
            "neatlogs.llm.token_count.completion", _get("neatlogs.raw.gen_ai.usage.output_tokens")
        )
        _set_if_missing(
            "neatlogs.llm.token_count.total", _get("neatlogs.raw.llm.usage.total_tokens")
        )

        _set_if_missing(
            "neatlogs.llm.token_count.cache_read",
            _get("neatlogs.raw.gen_ai.usage.cache_read_input_tokens"),
        )
        _set_if_missing(
            "neatlogs.llm.token_count.cache_write",
            _get("neatlogs.raw.gen_ai.usage.cache_creation_input_tokens"),
        )

        ca.pop("neatlogs.raw.llm.usage.total_tokens", None)

        canonical["attributes"] = ca

        we = wrapper.get("events") or []
        if we:
            ce = canonical.get("events") or []
            existing = {(e.get("name"), e.get("timestamp")) for e in ce}
            for e in we:
                key = (e.get("name"), e.get("timestamp"))
                if key not in existing:
                    ce.append(e)
                    existing.add(key)
            if any(
                e.get("name")
                in {
                    "llm.content.completion.chunk",
                    "gen_ai.content.completion.chunk",
                    "neatlogs.gen_ai.content.completion.chunk",
                }
                for e in ce
            ):
                ce = [e for e in ce if e.get("name") != "First Token Stream Event"]
            canonical["events"] = ce

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            self._flush_ready_spans(force=True)

            self.exporter.flush(timeout=timeout_millis / 1000.0)
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        try:
            self._stop_background.set()
            self._pending_event.set()
            if self._background_thread.is_alive():
                self._background_thread.join(timeout=1.0)
            self.force_flush()
        finally:
            self._log_performance_stats()
            self.exporter.shutdown()
            if self._raw_log_file_handle:
                try:
                    self._raw_log_file_handle.close()
                except Exception as e:
                    logger.warning(f"Failed to close raw log file handle: {e}")

    def _log_performance_stats(self) -> None:
        if not self.debug:
            return
        stats = self.perf_stats
        if stats["spans_processed"] == 0:
            return
        total_time = stats["on_start_time"] + stats["on_end_time"]
        avg_ms = (total_time / stats["spans_processed"]) * 1000
        try:
            logger.info(
                f"Neatlogs overhead: {total_time * 1000:.2f}ms total, {avg_ms:.3f}ms/span "
                f"({stats['spans_processed']} spans, "
                f"pending high watermark: {self._pending_high_watermark}, "
                f"total dropped: {self._pending_dropped})"
            )
        except (ValueError, OSError):
            pass
