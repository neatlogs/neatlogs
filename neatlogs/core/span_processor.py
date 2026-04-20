"""
Neatlogs span processor — attribute normalization and file logging.
Transport is handled by BatchSpanProcessor + OTLPSpanExporter (added in init.py).
"""

import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from .attribute_processor import UnifiedAttributeProcessor
from .logger import get_logger
from .mask import apply_mask

logger = get_logger()


class NeatlogsSpanProcessor(SpanProcessor):
    def __init__(
        self,
        sample_rate: float = 1.0,
        debug: bool = False,
        mask: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.sample_rate = sample_rate
        self.debug = debug
        self.mask = mask

        self._init_processor()
        self._init_file_logging()

        self.perf_stats = {
            "on_start_time": 0.0,
            "on_end_time": 0.0,
            "spans_processed": 0,
            "spans_exported": 0,
        }
        # Track parent span IDs scheduled for suppression (RETRIEVER dedup)
        self._retrievers_to_suppress: set = set()

    def _init_processor(self) -> None:
        base_path = os.path.dirname(os.path.dirname(__file__))
        mapping_path = os.path.join(base_path, "config", "attribute-mapping.json")
        try:
            with open(mapping_path, "r") as f:
                mapping_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load attribute-mapping.json: {e}")
            mapping_config = {}
        self.unified_processor = UnifiedAttributeProcessor(
            mapping_config=mapping_config,
            debug=self.debug,
        )

    def _init_file_logging(self) -> None:
        # Raw OTel span JSON (before attribute normalization)
        self._log_raw_spans_enabled = self.debug or (
            os.getenv("NEATLOGS_LOG_RAW_SPANS", "").lower() in ["true", "1", "yes"]
        )
        self._raw_log_file_handle = None
        if self._log_raw_spans_enabled:
            raw_path = os.path.join(
                os.getcwd(), os.getenv("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_optimized.log")
            )
            try:
                self._raw_log_file_handle = open(raw_path, "a", encoding="utf-8")
            except Exception:
                self._raw_log_file_handle = None

        # Processed span dict (after normalization — human-readable JSON lines)
        self._log_processed_spans_enabled = os.getenv("NEATLOGS_LOG_SPANS", "").lower() in [
            "true",
            "1",
            "yes",
        ]
        self._processed_log_file_handle = None
        if self._log_processed_spans_enabled:
            processed_path = os.path.join(
                os.getcwd(), os.getenv("NEATLOGS_LOG_SPANS_FILE", "spans_optimized.log")
            )
            try:
                self._processed_log_file_handle = open(processed_path, "a", encoding="utf-8")
                logger.info(f"Processed span logging enabled: {processed_path}")
            except Exception:
                self._processed_log_file_handle = None

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
        # Skip processing for internal completion markers — they only need to
        # be exported as-is by the downstream BatchSpanProcessor.
        if span.name == "neatlogs.trace.complete":
            return

        start_time = time.perf_counter()
        self.perf_stats["spans_processed"] += 1

        try:
            if self.debug:
                logger.debug(f"[SpanProcessor.on_end] Span ending: {span.name}")

            # 1. Log raw OTel span (before any processing)
            if self._raw_log_file_handle and not self._raw_log_file_handle.closed:
                try:
                    self._raw_log_file_handle.write(span.to_json() + "\n")
                    self._raw_log_file_handle.flush()
                except Exception as e:
                    logger.warning(f"Failed to write span to raw log file: {e}")

            if self.sample_rate < 1.0 and random.random() > self.sample_rate:
                return

            # 2. Process and normalize attributes
            unified_attrs = self.unified_processor.process(span)

            # 3. Filter large tokenized arrays for EMBEDDING/VECTOR_STORE spans
            nl_kind = unified_attrs.get("neatlogs.span.kind")
            if nl_kind in ("embedding", "vector_store"):
                skip_output = unified_attrs.get("neatlogs._skip_output_value") == True

                keys_to_remove = []
                for key in unified_attrs.keys():
                    if (
                        "input_messages" in key
                        or "output_messages" in key
                        or "gen_ai.prompt" in key
                        or "gen_ai.completion" in key
                        or ".content" in key
                    ):
                        keys_to_remove.append(key)
                    elif skip_output and (
                        key == "neatlogs.embedding.input"
                        or key == "neatlogs.embedding.output"
                    ):
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    unified_attrs.pop(key, None)

                if self.debug and keys_to_remove:
                    logger.debug(
                        f"[EMBEDDING Filter] Removed {len(keys_to_remove)} large attribute keys "
                        f"from {nl_kind} span (skip_output={skip_output})"
                    )

            # 4. Filter prompt template keys for non-LLM spans
            nl_kind = unified_attrs.get("neatlogs.span.kind")
            if nl_kind not in ("llm", "embedding", "crewai_task") and span.name != "PromptTemplate":
                for k in (
                    "neatlogs.llm.prompt_template",
                    "neatlogs.llm.prompt_template_variables",
                    "neatlogs.llm.prompt_template.version",
                ):
                    unified_attrs.pop(k, None)

            if span.name == "PromptTemplate":
                unified_attrs.setdefault("neatlogs.internal", True)
                unified_attrs["neatlogs.span.kind"] = "Neatlogs.INTERNAL"

            # 4b. Retriever span dedup: when a neatlogs RETRIEVER ends inside an OI
            # RETRIEVER parent, schedule the OI parent for suppression.
            # Then when the OI parent ends, mark it as internal.
            if nl_kind == "retriever":
                is_internal = unified_attrs.get("neatlogs.internal", False)
                # When a neatlogs (internal) RETRIEVER ends, schedule its OI parent
                # for suppression. OI attributes are set after span start so we can't
                # track parent type in on_start — instead the nl_kind == "retriever"
                # gate below ensures only RETRIEVER parents actually get suppressed.
                if is_internal and span.parent:
                    self._retrievers_to_suppress.add(span.parent.span_id)
                if span.context.span_id in self._retrievers_to_suppress:
                    self._retrievers_to_suppress.discard(span.context.span_id)
                    unified_attrs["neatlogs.internal"] = True
                    # Do NOT change neatlogs.span.kind — "retriever" must be preserved
                    # so the backend stores span_type = "RETRIEVER" correctly.
                    if self.debug:
                        logger.debug(
                            f"[Retriever Merge] Marked OI retriever '{span.name}' as internal "
                            f"(had neatlogs retriever child)"
                        )

            # 5. Build resource attributes
            resource_attrs = {}
            if span.resource and span.resource.attributes:
                for key, value in span.resource.attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        resource_attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        resource_attrs[key] = list(value)
                    else:
                        resource_attrs[key] = str(value)

                if self.debug and "neatlogs.tags" in resource_attrs:
                    logger.debug(
                        f"[Tags] Span {span.name}: resource.neatlogs.tags = {resource_attrs['neatlogs.tags']}"
                    )

            trace_id = f"{span.context.trace_id:032x}"
            span_id = f"{span.context.span_id:016x}"
            parent_span_id = f"{span.parent.span_id:016x}" if span.parent else None

            if parent_span_id == span_id:
                if self.debug:
                    logger.warning(
                        "[SpanProcessor] Detected self-parenting span. "
                        f"trace_id={trace_id} span_id={span_id} name={span.name}. "
                        "Setting parent_span_id=None."
                    )
                parent_span_id = None

            # 6. Build span_data dict (used for file logging)
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
                "resource": {"attributes": resource_attrs},
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
                    ]
                    if span.events
                    else []
                ),
            }

            # 7. Per-span post-processing (framework span name normalization, CrewAI tasks)
            results = self._normalize_framework_span_names([span_data])
            span_data = results[0] if results else span_data
            results = self._inject_crewai_task_templates([span_data])
            span_data = results[0] if results else span_data

            # 7b. Apply client-side mask before writing attributes back.
            # This ensures masked values flow to both OTLP export and file logging.
            if self.mask is not None:
                span_data = apply_mask(span_data, self.mask)

            # 7c. Write normalized (and masked) attributes back to the OTel span so that
            # BatchSpanProcessor → OTLPSpanExporter exports them to the backend.
            # ReadableSpan._attributes is a BoundedAttributes (MutableMapping) and is
            # mutable even after span.end(). The original OI attributes stay on the span
            # as-is; we only add the normalized neatlogs.* keys alongside them.
            final_attrs = span_data.get("attributes") or {}
            try:
                span_attrs = span._attributes
                if span_attrs is not None:
                    for _k, _v in final_attrs.items():
                        if isinstance(_v, (str, int, float, bool)):
                            span_attrs[_k] = _v
                        elif isinstance(_v, (list, tuple)) and all(
                            isinstance(_i, (str, int, float, bool)) for _i in _v
                        ):
                            span_attrs[_k] = list(_v)
                    # Unmapped attributes (e.g. input.user_email) stay on the OTel
                    # span outside the neatlogs.* namespace. Apply the mask to those
                    # too by building a temporary dict keyed by the original attr
                    # names, running the mask, and writing redacted values back.
                    if self.mask is not None:
                        raw_span = {
                            "name": span_data.get("name"),
                            "attributes": dict(span_attrs),
                        }
                        masked_raw = apply_mask(raw_span, self.mask)
                        masked_raw_attrs = masked_raw.get("attributes") or {}
                        for _k in list(span_attrs.keys()):
                            if _k in masked_raw_attrs and masked_raw_attrs[_k] != span_attrs[_k]:
                                span_attrs[_k] = masked_raw_attrs[_k]
            except Exception as _wb_exc:
                if self.debug:
                    logger.debug(f"[SpanProcessor] Attr write-back failed: {_wb_exc}")

            # 8. Log processed span dict (human-readable JSON lines, same format as before)
            if self._processed_log_file_handle and not self._processed_log_file_handle.closed:
                try:
                    self._processed_log_file_handle.write(
                        json.dumps(span_data) + "\n"
                    )
                    self._processed_log_file_handle.flush()
                except Exception as e:
                    logger.warning(f"Failed to write span to processed log file: {e}")

            self.perf_stats["spans_exported"] += 1

            # Emit a completion marker when a root span ends so the backend
            # knows the trace is complete and can trigger simplification.
            if not span.parent:
                self._emit_completion_marker(span, trace_id, resource_attrs)

        finally:
            self.perf_stats["on_end_time"] += time.perf_counter() - start_time

    def _emit_completion_marker(
        self,
        root_span: ReadableSpan,
        trace_id: str,
        resource_attrs: dict,
    ) -> None:
        """Emit a neatlogs.trace.complete span so the backend triggers simplification."""
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

            # Build a context that sits in the same trace but has no parent,
            # matching the old SDK's completion_marker behaviour.
            span_ctx = SpanContext(
                trace_id=root_span.context.trace_id,
                span_id=root_span.context.span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            ctx = otel_trace.set_span_in_context(NonRecordingSpan(span_ctx))
            tracer = otel_trace.get_tracer("neatlogs.internal")
            marker = tracer.start_span("neatlogs.trace.complete", context=ctx)
            marker.set_attribute("neatlogs.trace.complete", True)
            marker.set_attribute("neatlogs.internal", True)
            marker.set_attribute("neatlogs.span.kind", "Neatlogs.INTERNAL")
            # Copy resource tags so the backend can route the marker correctly.
            if resource_attrs.get("neatlogs.tags"):
                marker.set_attribute("neatlogs.tags", resource_attrs["neatlogs.tags"])
            marker.end()
            if self.debug:
                logger.debug(f"[SpanProcessor] Emitted completion marker for trace {trace_id}")
        except Exception as e:
            logger.warning(f"[SpanProcessor] Failed to emit completion marker: {e}")

    def _normalize_framework_span_names(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for s in spans:
            name = s.get("name") or ""
            kind = s.get("kind") or (s.get("attributes") or {}).get("neatlogs.span.kind")
            if kind != "task" or not name.endswith(".task"):
                continue

            attrs = s.get("attributes") or {}
            if not any(k.startswith("neatlogs.crewai.") for k in attrs.keys()):
                continue

            desc = name[: -len(".task")].rstrip()
            while desc.endswith("."):
                desc = desc[:-1].rstrip()

            if desc:
                attrs.setdefault("neatlogs.task.description", desc)
            s["name"] = "crewai.task"
            s["attributes"] = attrs

        return spans

    def _inject_crewai_task_templates(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from .crewai_task_registry import pop_entry

        for s in spans:
            attrs = s.get("attributes") or {}
            task_id = attrs.get("neatlogs.task.id")
            if not task_id:
                continue
            entry = pop_entry(str(task_id))
            if not entry:
                continue
            tpl_str, vars_json = entry
            attrs["neatlogs.task.user_prompt_template"] = tpl_str
            if vars_json:
                attrs["neatlogs.task.user_prompt_template_variables"] = vars_json
            attrs["neatlogs.span.kind"] = "crewai_task"
            s["attributes"] = attrs

        return spans

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # Flushing is handled by BatchSpanProcessor downstream.
        return True

    def shutdown(self) -> None:
        self._log_performance_stats()
        if self._raw_log_file_handle:
            try:
                self._raw_log_file_handle.close()
            except Exception as e:
                logger.warning(f"Failed to close raw log file handle: {e}")
        if self._processed_log_file_handle:
            try:
                self._processed_log_file_handle.close()
            except Exception as e:
                logger.warning(f"Failed to close processed log file handle: {e}")

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
                f"({stats['spans_processed']} spans processed, "
                f"{stats['spans_exported']} spans logged)"
            )
        except (ValueError, OSError):
            pass
