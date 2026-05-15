"""
Instrumentation manager.
"""

import importlib
import json
import logging
from functools import wraps
from typing import List, Optional, Set

from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from .http_context_propagation import patch_http_context_propagation
from .registry import INSTRUMENTATION_REGISTRY, get_libraries_by_tag

logger = logging.getLogger(__name__)


class InstrumentationManager:

    def __init__(
        self, provider: TracerProvider, debug: bool = False, excluded_urls: Optional[str] = None
    ):
        self.provider = provider
        self.debug = debug
        self.excluded_urls = excluded_urls
        self.instrumented: Set[str] = set()

    def instrument_threading(self) -> None:
        try:
            ThreadingInstrumentor().instrument()
            if self.debug:
                logger.info("✅ Instrumented threading (context propagation)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to instrument threading: {e}")

    def instrument_http(self) -> None:
        """
        Instrument HTTP libraries for context propagation.
        Uses standard opentelemetry-instrumentation-* contrib packages (not AI-specific).
        """
        http_libs = ["requests", "httpx", "urllib3", "aiohttp"]

        for lib in http_libs:
            if not self._is_library_installed(lib):
                if self.debug:
                    logger.info(f"⏭️  Skipped HTTP: {lib} (not installed)")
                continue

            try:
                self._instrument_library(lib, convention="openllmetry")
                self.instrumented.add(lib)
                if self.debug:
                    logger.info(f"✅ Instrumented HTTP: {lib}")
            except Exception as e:
                if self.debug:
                    logger.warning(f"⚠️  Failed to instrument {lib}: {e}")

        try:
            patch_http_context_propagation()
            if self.debug:
                logger.info("✅ Patched HTTP context propagation (best-effort)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch HTTP context propagation: {e}")

    def instrument_mcp(self) -> None:
        """
        Instrument MCP for cross-process context propagation.
        Uses OpenInference only.
        """
        if not self._is_library_installed("mcp"):
            if self.debug:
                logger.info("⏭️  Skipped MCP: not installed")
            return

        try:
            from openinference.instrumentation.mcp import MCPInstrumentor

            MCPInstrumentor().instrument(tracer_provider=self.provider)
            self.instrumented.add("mcp")
            if self.debug:
                logger.info("✅ MCP (OpenInference)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  MCP (OpenInference): {e}")

    def instrument(
        self, tags: Optional[List[str]] = None, libraries: Optional[List[str]] = None
    ) -> None:
        tags = tags or []
        libraries = libraries or []
        tag_libraries = set()
        for tag in tags:
            tag_libraries.update(get_libraries_by_tag(tag))
        all_libraries = tag_libraries.union(set(libraries))
        for lib in list(all_libraries):
            info = INSTRUMENTATION_REGISTRY["libraries"].get(lib, {})
            all_libraries.update(info.get("auto_load", []))
        for lib in all_libraries:
            if lib in self.instrumented:
                continue
            self._instrument_dual(lib)

    def _instrument_dual(self, library: str) -> None:
        """
        Instrument a library using priority order:
          1. Neatlogs custom instrumentor (if available)
          2. OpenInference (primary — rich semantic attributes, no duplicates)
          3. Stop — no OpenLLMetry fallback for AI libraries
        """
        if not self._is_library_installed(library):
            if self.debug:
                logger.info(f"⏭️  Skipped: {library} (not installed)")
            return

        info = INSTRUMENTATION_REGISTRY["libraries"].get(library)
        if not info:
            if self.debug:
                logger.warning(f"⚠️  Unknown library: {library}")
            return

        # 1. Neatlogs custom instrumentor (highest priority)
        if info.get("neatlogs"):
            try:
                self._instrument_library(library, convention="neatlogs")
                self.instrumented.add(library)
                if self.debug:
                    logger.info(f"✅ {library} (Neatlogs - custom)")
                return
            except Exception as e:
                if self.debug:
                    logger.warning(f"⚠️  {library} (Neatlogs): {e}")

        # 2. OpenInference (primary convention — no OpenLLMetry fallback)
        if info.get("openinference"):
            try:
                self._instrument_library(library, convention="openinference")
                self.instrumented.add(library)
                if self.debug:
                    logger.info(f"✅ {library} (OpenInference)")
                return
            except Exception as e:
                if self.debug:
                    logger.warning(f"⚠️  {library} (OpenInference): {e}")

        if self.debug:
            logger.info(f"⏭️  {library}: no instrumentor available (skipped)")

    def _instrument_library(self, library: str, convention: str) -> None:
        info = INSTRUMENTATION_REGISTRY["libraries"][library]
        package_name = info.get(convention)

        if not package_name:
            return

        try:
            module = importlib.import_module(package_name)

            instrumentor_class_name = self._get_instrumentor_class_name(library, convention)
            instrumentor_class = getattr(module, instrumentor_class_name)

            is_http_lib = library in ["requests", "httpx", "urllib3", "aiohttp"]
            if is_http_lib and self.excluded_urls:
                instrumentor_class().instrument(
                    tracer_provider=self.provider, excluded_urls=self.excluded_urls
                )
            else:
                instrumentor_class().instrument(tracer_provider=self.provider)

            # Post-instrument patches for OpenInference libraries
            if convention == "openinference":
                if library == "openai":
                    self._patch_openinference_openai_request_extras()
                    self._patch_openinference_openai_response_extras()
                    self._patch_openinference_openai_streaming_timing()
                    self._patch_openai_streaming_usage()
                elif library == "anthropic":
                    self._patch_openinference_anthropic_response_extras()
                    self._patch_openinference_anthropic_streaming_timing()
                elif library == "google_genai":
                    self._patch_openinference_google_genai_stream_finally()
                    self._patch_openinference_google_genai_response_extras()
                elif library == "langchain":
                    self._patch_openinference_langchain_finish_reason()
                    self._patch_openinference_langchain_streaming_timing()
                    self._patch_openinference_langchain_suppress_internal()
                    self._patch_openinference_langchain_suppress_internal()
                elif library == "litellm":
                    self._patch_openinference_litellm_ignore_instrumentation_suppression()
                    self._patch_openinference_litellm_streaming_timing()
                elif library == "crewai":
                    self._patch_openinference_crewai_crew_outputs()
                    self._patch_crewai_tool_spans()

        except Exception as e:
            raise Exception(f"Failed to instrument {library} with {convention}: {e}")

    # ---------------------------------------------------------------------------
    # OpenInference — OpenAI patches
    # ---------------------------------------------------------------------------

    def _patch_openinference_openai_request_extras(self) -> None:
        """
        Patch OI OpenAI request extractor to capture attributes OI omits:
          - llm.is_streaming  (from `stream`)
          - llm.user          (from `user`)
          - llm.headers       (from `extra_headers` / `headers`)
          - llm.reasoning_effort (from `reasoning_effort`)
          - llm.request.structured_output_schema (from `response_format`)
        """
        try:
            from openinference.instrumentation.openai._request_attributes_extractor import (
                _RequestAttributesExtractor,
            )

            if getattr(_RequestAttributesExtractor, "_NEATLOGS_PATCHED_REQUEST_EXTRAS", False):
                return

            original = _RequestAttributesExtractor._get_attributes_from_chat_completion_create_param

            def _patched(self_ext, request_parameters):
                yield from original(self_ext, request_parameters)

                if stream := request_parameters.get("stream"):
                    yield "llm.is_streaming", bool(stream)
                if user := request_parameters.get("user"):
                    yield "llm.user", str(user)
                headers = request_parameters.get("extra_headers") or request_parameters.get(
                    "headers"
                )
                if headers:
                    yield "llm.headers", str(headers)
                if reasoning_effort := request_parameters.get("reasoning_effort"):
                    yield "llm.reasoning_effort", str(reasoning_effort)

                response_format = request_parameters.get("response_format")
                if response_format:
                    schema_json = _extract_structured_output_schema(response_format)
                    if schema_json:
                        yield "llm.request.structured_output_schema", schema_json

            _RequestAttributesExtractor._get_attributes_from_chat_completion_create_param = _patched
            _RequestAttributesExtractor._NEATLOGS_PATCHED_REQUEST_EXTRAS = True

            if self.debug:
                logger.debug("Patched OI OpenAI: request extras (streaming/user/headers/schema)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI OpenAI request extras: {e}")

    def _patch_openinference_openai_response_extras(self) -> None:
        """
        Patch OI OpenAI response extractor to capture attributes OI omits:
          - llm.output_messages.{i}.message.finish_reason (from choice.finish_reason)
          - llm.output_messages.{i}.message.refusal       (from choice.message.refusal)
        """
        try:
            from openinference.instrumentation.openai._response_attributes_extractor import (
                _ResponseAttributesExtractor,
            )

            if getattr(_ResponseAttributesExtractor, "_NEATLOGS_PATCHED_RESPONSE_EXTRAS", False):
                return

            original = _ResponseAttributesExtractor._get_attributes_from_chat_completion

            def _patched(self_ext, completion, request_parameters):
                yield from original(self_ext, completion, request_parameters)

                choices = getattr(completion, "choices", None) or []
                for choice in choices:
                    index = getattr(choice, "index", None)
                    if index is None:
                        continue
                    prefix = f"llm.output_messages.{index}.message"
                    if finish_reason := getattr(choice, "finish_reason", None):
                        yield f"{prefix}.finish_reason", finish_reason
                    if message := getattr(choice, "message", None):
                        if refusal := getattr(message, "refusal", None):
                            yield f"{prefix}.refusal", str(refusal)

            _ResponseAttributesExtractor._get_attributes_from_chat_completion = _patched
            _ResponseAttributesExtractor._NEATLOGS_PATCHED_RESPONSE_EXTRAS = True

            if self.debug:
                logger.debug("Patched OI OpenAI: response extras (finish_reason/refusal)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI OpenAI response extras: {e}")

    # ---------------------------------------------------------------------------
    # OpenInference — Anthropic patches
    # ---------------------------------------------------------------------------

    def _patch_openinference_anthropic_response_extras(self) -> None:
        """
        Patch OI Anthropic response processing to capture attributes OI omits:
          - llm.output_messages.0.message.finish_reason  (from response.stop_reason)
          - llm.output_messages.{n}.message.role/content  (thinking blocks as separate messages)
        """
        try:
            import openinference.instrumentation.anthropic._wrappers as _ant

            if getattr(_ant, "_NEATLOGS_PATCHED_RESPONSE_EXTRAS", False):
                return

            original_get_output_messages = _ant._get_output_messages
            LLM_OUTPUT_MESSAGES = _ant.LLM_OUTPUT_MESSAGES

            def _patched_get_output_messages(response):
                yield from original_get_output_messages(response)

                # stop_reason → finish_reason on message 0
                if stop_reason := getattr(response, "stop_reason", None):
                    yield f"{LLM_OUTPUT_MESSAGES}.0.message.finish_reason", stop_reason

                # Thinking blocks → additional output message slots
                try:
                    from anthropic.types import ThinkingBlock

                    thinking_index = 1
                    for block in getattr(response, "content", None) or []:
                        if isinstance(block, ThinkingBlock) and block.thinking:
                            yield f"{LLM_OUTPUT_MESSAGES}.{thinking_index}.message.role", "thinking"
                            yield f"{LLM_OUTPUT_MESSAGES}.{thinking_index}.message.content", block.thinking
                            thinking_index += 1
                except ImportError:
                    pass

            _ant._get_output_messages = _patched_get_output_messages
            _ant._NEATLOGS_PATCHED_RESPONSE_EXTRAS = True

            if self.debug:
                logger.debug("Patched OI Anthropic: response extras (stop_reason/thinking blocks)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI Anthropic response extras: {e}")

    # ---------------------------------------------------------------------------
    # OpenInference — Google GenAI patches
    # ---------------------------------------------------------------------------

    def _patch_openinference_google_genai_stream_finally(self) -> None:
        """
        Patch OI google_genai _Stream.__aiter__ so async spans always end.

        The upstream __aiter__ only calls _finish_tracing on natural exhaustion or
        exception. If the async generator is abandoned early (caller breaks, discards
        the stream, or GC collects without aclose()), the span is never ended.
        """
        try:
            from openinference.instrumentation.google_genai._stream import _Stream
            from opentelemetry import trace as trace_api

            async def _fixed_aiter(self):
                status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                try:
                    async for item in self.__wrapped__:
                        try:
                            chunk_text = ""
                            if hasattr(item, "text") and item.text:
                                chunk_text = item.text
                            elif hasattr(item, "candidates") and item.candidates:
                                for candidate in item.candidates:
                                    if hasattr(candidate, "content") and candidate.content:
                                        if (
                                            hasattr(candidate.content, "parts")
                                            and candidate.content.parts
                                        ):
                                            for part in candidate.content.parts:
                                                if hasattr(part, "text") and part.text:
                                                    chunk_text += part.text
                            if chunk_text:
                                self._with_span.span.add_event(
                                    name="gen_ai.content.chunk",
                                    attributes={
                                        "gen_ai.content.chunk.index": self._chunk_index,
                                        "gen_ai.content.chunk.text": chunk_text[:500],
                                    },
                                )
                                self._chunk_index += 1
                        except Exception:
                            pass
                        self._response_accumulator.process_chunk(item)
                        yield item
                except Exception as exception:
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    self._with_span.record_exception(exception)
                    raise
                finally:
                    self._finish_tracing(status=status)

            _Stream.__aiter__ = _fixed_aiter

            if self.debug:
                logger.debug("Patched OI google_genai _Stream: try/finally async span completion")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI google_genai _Stream: {e}")

    def _patch_openinference_google_genai_response_extras(self) -> None:
        """
        Patch OI google_genai response extractor to capture attributes OI omits:
          - llm.output_messages.{i}.message.finish_reason  (from candidate.finish_reason)
          - llm.output_messages.{i}.message.safety_ratings (from candidate.safety_ratings)
          - Separate thinking parts into role="thinking" output message
        """
        try:
            from openinference.instrumentation.google_genai._response_attributes_extractor import (
                _ResponseAttributesExtractor,
            )

            if getattr(_ResponseAttributesExtractor, "_NEATLOGS_PATCHED_RESPONSE_EXTRAS", False):
                return

            original = _ResponseAttributesExtractor._get_attributes_from_generate_content

            def _patched(self_ext, response, **kwargs):
                yield from original(self_ext, response, **kwargs)

                candidates = getattr(response, "candidates", None) or []
                for i, candidate in enumerate(candidates):
                    if finish_reason := getattr(candidate, "finish_reason", None):
                        finish_str = (
                            finish_reason.name
                            if hasattr(finish_reason, "name")
                            else str(finish_reason)
                        )
                        yield f"llm.output_messages.{i}.message.finish_reason", finish_str

                    safety_ratings = getattr(candidate, "safety_ratings", None)
                    if safety_ratings:
                        try:
                            ratings_list = [
                                {
                                    "category": (
                                        r.category.name
                                        if hasattr(r.category, "name")
                                        else str(r.category)
                                    ),
                                    "probability": (
                                        r.probability.name
                                        if hasattr(r.probability, "name")
                                        else str(r.probability)
                                    ),
                                }
                                for r in safety_ratings
                            ]
                            yield f"llm.output_messages.{i}.message.safety_ratings", json.dumps(
                                ratings_list
                            )
                        except Exception:
                            pass

                    # Separate thinking parts from response parts
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None) or []
                    thinking_texts = []
                    for part in parts:
                        thought = getattr(part, "thought", False)
                        text = getattr(part, "text", None)
                        if thought and text:
                            thinking_texts.append(text)
                    if thinking_texts:
                        thinking_slot = len(candidates) + i
                        yield f"llm.output_messages.{thinking_slot}.message.role", "thinking"
                        yield f"llm.output_messages.{thinking_slot}.message.content", "\n".join(
                            thinking_texts
                        )

            _ResponseAttributesExtractor._get_attributes_from_generate_content = _patched
            _ResponseAttributesExtractor._NEATLOGS_PATCHED_RESPONSE_EXTRAS = True

            if self.debug:
                logger.debug(
                    "Patched OI google_genai: response extras (finish_reason/safety_ratings/thinking)"
                )
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI google_genai response extras: {e}")

    # ---------------------------------------------------------------------------
    # OpenInference — LangChain patches
    # ---------------------------------------------------------------------------

    def _patch_openinference_langchain_finish_reason(self) -> None:
        """
        Patch OI LangChain tracer to extract finish_reason from generation_info
        and stamp it on output message spans.
        """
        try:
            import openinference.instrumentation.langchain._tracer as _lc_tracer

            if getattr(_lc_tracer, "_NEATLOGS_PATCHED_FINISH_REASON", False):
                return

            original_update_span = _lc_tracer._update_span

            def _patched_update_span(span, run):
                original_update_span(span, run)

                if run.run_type != "llm" or not run.outputs:
                    return
                try:
                    generations = run.outputs.get("generations") or []
                    for gen_list in generations:
                        for i, gen in enumerate(gen_list or []):
                            if not hasattr(gen, "get"):
                                continue
                            gen_info = gen.get("generation_info") or {}
                            finish_reason = (
                                gen_info.get("finish_reason")
                                or gen_info.get("stop_reason")
                                or gen_info.get("finishReason")
                            )
                            if finish_reason:
                                span.set_attribute(
                                    f"llm.output_messages.{i}.message.finish_reason",
                                    str(finish_reason),
                                )
                except Exception:
                    pass

            _lc_tracer._update_span = _patched_update_span
            _lc_tracer._NEATLOGS_PATCHED_FINISH_REASON = True

            if self.debug:
                logger.debug("Patched OI LangChain: finish_reason from generation_info")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI LangChain finish_reason: {e}")

    def _patch_openinference_langchain_suppress_internal(self) -> None:
        """
        Suppress LCEL plumbing spans (PromptTemplate, RunnableSequence,
        RunnableLambda, parsers, etc.) and transparently reparent their
        children to the nearest kept ancestor.

        When a run is suppressed in _start_trace, we map its run_id to its
        parent's span in _spans_by_run so any child looking up parent_run_id
        transparently finds the grandparent's span as context.
        """
        try:
            from openinference.instrumentation.langchain._tracer import OpenInferenceTracer
            from opentelemetry import context as context_api
            from opentelemetry import trace as trace_api

            if getattr(OpenInferenceTracer, "_NEATLOGS_PATCHED_SUPPRESS_INTERNAL", False):
                return

            _SUPPRESS_EXACT = frozenset(
                {
                    "PromptTemplate",
                    "ChatPromptTemplate",
                    "RunnableLambda",
                    "RunnableSequence",
                    "ReActSingleInputOutputParser",
                    "StrOutputParser",
                    "JsonOutputParser",
                    "XMLOutputParser",
                }
            )
            _SUPPRESS_PREFIXES = ("RunnableAssign", "RunnableParallel")

            def _should_suppress(name):
                if not name:
                    return False
                return name in _SUPPRESS_EXACT or any(
                    name.startswith(p) for p in _SUPPRESS_PREFIXES
                )

            _orig_start_trace = OpenInferenceTracer._start_trace

            def _patched_start_trace(self, run):
                if _should_suppress(run.name):
                    if not hasattr(self, "_nl_skipped_runs"):
                        self._nl_skipped_runs = set()
                    self._nl_skipped_runs.add(run.id)
                    # Keep run indexed so BaseTracer.on_chain_end doesn't throw
                    self.run_map[str(run.id)] = run
                    # Map this run_id → parent's span so children reparent up.
                    # Priority: LangChain parent span → ambient OTel context span.
                    # The OTel fallback handles the case where the suppressed span
                    # has no LangChain parent (e.g. called from a neatlogs.trace block).
                    parent_span = None
                    if run.parent_run_id:
                        parent_span = self._spans_by_run.get(run.parent_run_id)
                    if parent_span is None:
                        current = trace_api.get_current_span()
                        if current is not None and current.is_recording():
                            parent_span = current
                    if parent_span is not None:
                        self._spans_by_run[run.id] = parent_span
                    return
                _orig_start_trace(self, run)
                # Attach the created span to thread-local OTel context so that
                # neatlogs.trace(), HTTP instrumentation, and other OTel spans
                # created inside LangChain callbacks see this span as parent.
                span = self._spans_by_run.get(run.id)
                if span is not None:
                    ctx = trace_api.set_span_in_context(span)
                    token = context_api.attach(ctx)
                    if not hasattr(self, "_nl_context_tokens"):
                        self._nl_context_tokens = {}
                    self._nl_context_tokens[run.id] = token

            # _end_trace may already be patched by streaming timing — chain on top
            _prev_end_trace = OpenInferenceTracer._end_trace

            def _patched_end_trace(self, run):
                skipped = getattr(self, "_nl_skipped_runs", None)
                if skipped and run.id in skipped:
                    skipped.discard(run.id)
                    self.run_map.pop(str(run.id), None)
                    self._spans_by_run.pop(run.id, None)
                    return
                # Detach thread-local context before ending the span
                tokens = getattr(self, "_nl_context_tokens", {})
                token = tokens.pop(run.id, None)
                if token is not None:
                    context_api.detach(token)
                _prev_end_trace(self, run)

            OpenInferenceTracer._start_trace = _patched_start_trace
            OpenInferenceTracer._end_trace = _patched_end_trace
            OpenInferenceTracer._NEATLOGS_PATCHED_SUPPRESS_INTERNAL = True

            if self.debug:
                logger.debug(
                    "Patched OI LangChain: suppress internal LCEL spans "
                    "(PromptTemplate, RunnableSequence, RunnableLambda, parsers)"
                )
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI LangChain suppress internal: {e}")

    # ---------------------------------------------------------------------------
    # OpenInference — LiteLLM patches
    # ---------------------------------------------------------------------------

    def _patch_openinference_litellm_ignore_instrumentation_suppression(self) -> None:
        try:
            import litellm
            from opentelemetry import context as context_api
            from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

            if getattr(litellm, "_NEATLOGS_PATCHED_IGNORE_OTEL_SUPPRESS", False):
                return

            def _wrap(fn):
                @wraps(fn)
                def _wrapped(*args, **kwargs):
                    token = None
                    try:
                        token = context_api.attach(
                            context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, False)
                        )
                    except Exception:
                        token = None
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        if token is not None:
                            try:
                                context_api.detach(token)
                            except Exception:
                                pass

                return _wrapped

            for name in (
                "completion",
                "acompletion",
                "responses",
                "aresponses",
                "completion_with_retries",
                "embedding",
                "aembedding",
                "image_generation",
                "aimage_generation",
            ):
                if hasattr(litellm, name):
                    fn = getattr(litellm, name)
                    if callable(fn):
                        setattr(litellm, name, _wrap(fn))

            litellm._NEATLOGS_PATCHED_IGNORE_OTEL_SUPPRESS = True
            if self.debug:
                logger.debug("Patched OI LiteLLM: ignore _SUPPRESS_INSTRUMENTATION_KEY")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI LiteLLM suppression: {e}")

    # ---------------------------------------------------------------------------
    # OpenInference — CrewAI patches
    # ---------------------------------------------------------------------------

    def _patch_openinference_crewai_crew_outputs(self) -> None:
        """
        Patch OI CrewAI Crew.kickoff wrapper to also capture:
          - neatlogs.crew.tasks_output  (per-task results from kickoff output)
          - neatlogs.crew.token_usage   (aggregated token usage from kickoff output)
        """
        try:
            import openinference.instrumentation.crewai._wrappers as _crew_wrappers

            if getattr(_crew_wrappers, "_NEATLOGS_PATCHED_CREW_OUTPUTS", False):
                return

            original_kickoff_call = _crew_wrappers._CrewKickoffWrapper.__call__

            def _patched_kickoff_call(self_wrapper, wrapped, instance, args, kwargs):
                result = original_kickoff_call(self_wrapper, wrapped, instance, args, kwargs)

                # Stamp extra attributes onto the current span if one is active
                try:
                    from opentelemetry import trace as _trace

                    span = _trace.get_current_span()
                    if span and span.is_recording():
                        if hasattr(result, "tasks_output") and result.tasks_output:
                            span.set_attribute(
                                "neatlogs.crew.tasks_output",
                                str(result.tasks_output),
                            )
                        token_usage = getattr(result, "token_usage", None) or getattr(
                            result, "usage_metrics", None
                        )
                        if token_usage:
                            span.set_attribute(
                                "neatlogs.crew.token_usage",
                                str(token_usage),
                            )
                except Exception:
                    pass

                return result

            _crew_wrappers._CrewKickoffWrapper.__call__ = _patched_kickoff_call
            _crew_wrappers._NEATLOGS_PATCHED_CREW_OUTPUTS = True

            if self.debug:
                logger.debug("Patched OI CrewAI: crew outputs (tasks_output/token_usage)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI CrewAI crew outputs: {e}")

    def _patch_crewai_tool_spans(self) -> None:
        """
        Patch CrewAgentExecutor._handle_native_tool_calls to wrap each tool
        execution with a TOOL span.

        CrewAI 1.9+ bypasses ToolUsage._use() and calls tool functions directly
        inside CrewAgentExecutor. The OI CrewAI instrumentor only patches
        ToolUsage._use, so tool spans are missing. This patch intercepts
        _handle_native_tool_calls and wraps the available_functions dict so
        each tool call gets a proper TOOL span.
        """
        try:
            from crewai.agents.crew_agent_executor import CrewAgentExecutor

            original = CrewAgentExecutor._handle_native_tool_calls
            tracer = self.provider.get_tracer("neatlogs.crewai.tools")

            def _make_tool_wrapper(fn, tool_name, _tracer):
                def _wrapped_tool(**kwargs):
                    import json as _json

                    from opentelemetry.trace import SpanKind as _SK
                    from opentelemetry.trace import StatusCode as _SC

                    with _tracer.start_as_current_span(
                        tool_name,
                        kind=_SK.INTERNAL,
                    ) as span:
                        span.set_attribute("openinference.span.kind", "TOOL")
                        span.set_attribute("tool.name", tool_name)
                        try:
                            span.set_attribute(
                                "input.value",
                                _json.dumps(kwargs, default=str),
                            )
                        except Exception:
                            pass
                        try:
                            result = fn(**kwargs)
                            try:
                                span.set_attribute("output.value", str(result))
                            except Exception:
                                pass
                            span.set_status(_SC.OK)
                            return result
                        except Exception as exc:
                            span.record_exception(exc)
                            span.set_status(_SC.ERROR)
                            raise

                return _wrapped_tool

            def _patched_handle(executor_self, tool_calls, available_functions):
                wrapped_fns = {
                    name: _make_tool_wrapper(func, name, tracer)
                    for name, func in available_functions.items()
                }
                return original(executor_self, tool_calls, wrapped_fns)

            CrewAgentExecutor._handle_native_tool_calls = _patched_handle

            if self.debug:
                logger.debug("Patched CrewAI: tool execution spans via CrewAgentExecutor")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch CrewAI tool spans: {e}")

    # ---------------------------------------------------------------------------
    # Streaming Timing Patches (TTFT + streaming_time_to_generate)
    # ---------------------------------------------------------------------------

    def _patch_openinference_openai_streaming_timing(self) -> None:
        """
        Patch OpenInference OpenAI _Stream to capture per-chunk timestamps.
        Sets neatlogs.llm.metrics.ttft_ms and streaming_time_to_generate_ms on the span
        before it closes, covering both sync (__next__) and async (__anext__) paths.
        """
        try:
            import time as _time

            from openinference.instrumentation.openai._stream import _Stream

            _orig_process_chunk = _Stream._process_chunk

            def _patched_process_chunk(self, chunk):
                if not hasattr(self, "_nl_timestamps"):
                    self._nl_timestamps = []
                self._nl_timestamps.append(_time.time())
                _orig_process_chunk(self, chunk)

            _orig_finish_tracing = _Stream._finish_tracing

            def _patched_finish_tracing(self, status=None):
                timestamps = getattr(self, "_nl_timestamps", [])
                if timestamps:
                    try:
                        span = self._self_with_span._span
                        first_ns = int(timestamps[0] * 1e9)
                        last_ns = int(timestamps[-1] * 1e9)
                        span.set_attribute(
                            "neatlogs.llm.metrics.ttft_ms",
                            round((first_ns - span.start_time) / 1_000_000, 3),
                        )
                        if last_ns > first_ns:
                            span.set_attribute(
                                "neatlogs.llm.metrics.streaming_time_to_generate_ms",
                                round((last_ns - first_ns) / 1_000_000, 3),
                            )
                    except Exception:
                        pass
                _orig_finish_tracing(self, status=status)

            _Stream._process_chunk = _patched_process_chunk
            _Stream._finish_tracing = _patched_finish_tracing

            if self.debug:
                logger.debug("Patched OI OpenAI: streaming timing (TTFT + time_to_generate)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI OpenAI streaming timing: {e}")

    def _patch_openai_streaming_usage(self) -> None:
        """
        Patch OpenAI client to inject stream_options={"include_usage": True}
        when stream=True, so the final streaming chunk includes token usage.
        OpenInference already extracts usage from the response — it just never
        receives it because OpenAI requires this opt-in flag for streaming.
        """
        try:
            import openai

            _patched_flag = "_NEATLOGS_PATCHED_STREAM_USAGE"
            if getattr(openai.resources.chat.completions.Completions, _patched_flag, False):
                return

            _orig_create = openai.resources.chat.completions.Completions.create

            def _patched_create(self_client, *args, **kwargs):
                if kwargs.get("stream"):
                    opts = kwargs.get("stream_options") or {}
                    if not opts.get("include_usage"):
                        opts["include_usage"] = True
                        kwargs["stream_options"] = opts
                return _orig_create(self_client, *args, **kwargs)

            openai.resources.chat.completions.Completions.create = _patched_create

            # Async variant
            _orig_acreate = openai.resources.chat.completions.AsyncCompletions.create

            async def _patched_acreate(self_client, *args, **kwargs):
                if kwargs.get("stream"):
                    opts = kwargs.get("stream_options") or {}
                    if not opts.get("include_usage"):
                        opts["include_usage"] = True
                        kwargs["stream_options"] = opts
                return await _orig_acreate(self_client, *args, **kwargs)

            openai.resources.chat.completions.AsyncCompletions.create = _patched_acreate

            setattr(openai.resources.chat.completions.Completions, _patched_flag, True)

            if self.debug:
                logger.debug("Patched OpenAI: inject stream_options.include_usage for streaming")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OpenAI streaming usage: {e}")

    def _patch_openinference_anthropic_streaming_timing(self) -> None:
        """
        Patch OpenInference Anthropic _MessageResponseAccumulator + _MessagesStream
        to capture per-chunk timestamps and set timing span attributes before span closes.
        Covers both sync and async streaming paths.
        """
        try:
            import time as _time

            from openinference.instrumentation.anthropic._stream import (
                _MessageResponseAccumulator,
                _MessagesStream,
            )

            # _MessageResponseAccumulator uses __slots__ with no __weakref__, so we
            # cannot set new instance attributes on it or use WeakKeyDictionary.
            # Use an external dict keyed by id(accumulator); cleaned up in _finish_tracing.
            _nl_chunk_timestamps: dict = {}

            _orig_process_chunk = _MessageResponseAccumulator.process_chunk

            def _patched_process_chunk(self, chunk):
                key = id(self)
                if key not in _nl_chunk_timestamps:
                    _nl_chunk_timestamps[key] = []
                _nl_chunk_timestamps[key].append(_time.time())
                _orig_process_chunk(self, chunk)

            _orig_finish_tracing = _MessagesStream._finish_tracing

            def _patched_finish_tracing(self, status=None):
                key = id(self._response_accumulator)
                timestamps = _nl_chunk_timestamps.pop(key, [])
                if timestamps:
                    try:
                        span = self._with_span._span
                        first_ns = int(timestamps[0] * 1e9)
                        last_ns = int(timestamps[-1] * 1e9)
                        span.set_attribute(
                            "neatlogs.llm.metrics.ttft_ms",
                            round((first_ns - span.start_time) / 1_000_000, 3),
                        )
                        if last_ns > first_ns:
                            span.set_attribute(
                                "neatlogs.llm.metrics.streaming_time_to_generate_ms",
                                round((last_ns - first_ns) / 1_000_000, 3),
                            )
                    except Exception:
                        pass
                _orig_finish_tracing(self, status=status)

            _MessageResponseAccumulator.process_chunk = _patched_process_chunk
            _MessagesStream._finish_tracing = _patched_finish_tracing

            if self.debug:
                logger.debug("Patched OI Anthropic: streaming timing (TTFT + time_to_generate)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI Anthropic streaming timing: {e}")

    def _patch_openinference_langchain_streaming_timing(self) -> None:
        """
        Patch OpenInference LangChain OpenInferenceTracer with on_llm_new_token
        to capture TTFT and streaming_time_to_generate for all LangChain-integrated
        LLMs (ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, etc.) uniformly.
        LangGraph flows through LangChain callbacks — covered by the same patch.
        """
        try:
            import time as _time

            from openinference.instrumentation.langchain._tracer import OpenInferenceTracer

            def _patched_on_llm_new_token(self, token, *, run_id, **kwargs):
                if not hasattr(self, "_nl_stream_times"):
                    self._nl_stream_times = {}
                now = _time.time()
                if run_id not in self._nl_stream_times:
                    self._nl_stream_times[run_id] = {"first": now}
                self._nl_stream_times[run_id]["last"] = now

            _orig_end_trace = OpenInferenceTracer._end_trace

            def _patched_end_trace(self, run):
                stream_data = getattr(self, "_nl_stream_times", {}).pop(run.id, None)
                if stream_data and (span := self._spans_by_run.get(run.id)):
                    try:
                        first_ns = int(stream_data["first"] * 1e9)
                        ttft_ms = round((first_ns - span.start_time) / 1_000_000, 3)
                        span.set_attribute("neatlogs.llm.metrics.ttft_ms", ttft_ms)
                        last = stream_data.get("last")
                        if last and last > stream_data["first"]:
                            stg_ms = round((last - stream_data["first"]) * 1000, 3)
                            span.set_attribute(
                                "neatlogs.llm.metrics.streaming_time_to_generate_ms", stg_ms
                            )
                    except Exception:
                        pass
                _orig_end_trace(self, run)

            OpenInferenceTracer.on_llm_new_token = _patched_on_llm_new_token
            OpenInferenceTracer._end_trace = _patched_end_trace

            if self.debug:
                logger.debug("Patched OI LangChain: streaming timing via on_llm_new_token")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI LangChain streaming timing: {e}")

    def _patch_openinference_litellm_streaming_timing(self) -> None:
        """
        Patch OpenInference LiteLLM _finalize_sync_streaming_span and
        _finalize_streaming_span to capture per-chunk timestamps and set
        timing span attributes before span.end() is called.
        Used by CrewAI (which routes all LLM calls through LiteLLM).
        """
        try:
            import time as _time

            import openinference.instrumentation.litellm as _litellm_module

            _orig_sync = _litellm_module._finalize_sync_streaming_span
            _orig_async = _litellm_module._finalize_streaming_span

            def _patched_finalize_sync(span, stream):
                class _T:
                    def __iter__(inner):
                        first = last = None
                        for token in stream:
                            now = _time.time()
                            if first is None:
                                first = now
                            last = now
                            yield token
                        if first is not None:
                            try:
                                span.set_attribute(
                                    "neatlogs.llm.metrics.ttft_ms",
                                    round((int(first * 1e9) - span.start_time) / 1_000_000, 3),
                                )
                                if last > first:
                                    span.set_attribute(
                                        "neatlogs.llm.metrics.streaming_time_to_generate_ms",
                                        round((last - first) * 1000, 3),
                                    )
                            except Exception:
                                pass

                    def __getattr__(inner, n):
                        return getattr(stream, n)

                return _orig_sync(span, _T())

            async def _patched_finalize_async(span, stream):
                class _T:
                    async def __aiter__(inner):
                        first = last = None
                        async for token in stream:
                            now = _time.time()
                            if first is None:
                                first = now
                            last = now
                            yield token
                        if first is not None:
                            try:
                                span.set_attribute(
                                    "neatlogs.llm.metrics.ttft_ms",
                                    round((int(first * 1e9) - span.start_time) / 1_000_000, 3),
                                )
                                if last > first:
                                    span.set_attribute(
                                        "neatlogs.llm.metrics.streaming_time_to_generate_ms",
                                        round((last - first) * 1000, 3),
                                    )
                            except Exception:
                                pass

                    def __getattr__(inner, n):
                        return getattr(stream, n)

                return _orig_async(span, _T())

            _litellm_module._finalize_sync_streaming_span = _patched_finalize_sync
            _litellm_module._finalize_streaming_span = _patched_finalize_async

            if self.debug:
                logger.debug("Patched OI LiteLLM: streaming timing (TTFT + time_to_generate)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OI LiteLLM streaming timing: {e}")

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _get_instrumentor_class_name(self, library: str, convention: str) -> str:
        if convention == "neatlogs":
            neatlogs_cases = {
                "openai": "OpenAIInstrumentor",
                "anthropic": "AnthropicInstrumentor",
                "langchain": "LangChainInstrumentor",
                "langgraph": "LangGraphInstrumentor",
                "crewai": "CrewAIInstrumentor",
                "bedrock": "BedrockInstrumentor",
                "groq": "GroqInstrumentor",
                "google_genai": "GoogleGenAIInstrumentor",
            }
            if library in neatlogs_cases:
                return neatlogs_cases[library]

        special_cases = {
            "openai": "OpenAIInstrumentor",
            "langchain": (
                "LangChainInstrumentor"
                if convention == "openinference"
                else "LangchainInstrumentor"
            ),
            "urllib3": "URLLib3Instrumentor",
            "httpx": "HTTPXClientInstrumentor",
            "aiohttp": "AioHttpClientInstrumentor",
            "llamaindex": "LlamaIndexInstrumentor",
            "google_generativeai": "GoogleGenerativeAIInstrumentor",
            "google_genai": "GoogleGenAIInstrumentor",
            "google_adk": "GoogleADKInstrumentor",
            "huggingface_hub": "HuggingfaceHubInstrumentor",
            "alephalpha": "AlephAlphaInstrumentor",
            "mistralai": "MistralAIInstrumentor",
            "vertexai": "VertexAIInstrumentor",
            "litellm": "LiteLLMInstrumentor",
            "crewai": "CrewAIInstrumentor",
            "azure_ai_inference": "AzureAIInferenceInstrumentor",
            "dspy": "DSPyInstrumentor",
            "chromadb": "ChromaInstrumentor",
            "beeai": "BeeAIInstrumentor",
            "openai_agents": "OpenAIAgentsInstrumentor",
            "pydantic_ai": "PydanticAIInstrumentor",
            "mcp": "MCPInstrumentor",
        }

        if library in special_cases:
            return special_cases[library]

        return f"{library.capitalize()}Instrumentor"

    def _is_library_installed(self, library: str) -> bool:
        try:
            special_imports = {
                "google_genai": "google.genai",
                "google_generativeai": "google.generativeai",
                "llamaindex": "llama_index",
                "azure_ai_inference": "azure.ai.inference",
                "bedrock": "boto3",
                "milvus": "pymilvus",
                "qdrant": "qdrant_client",
            }
            import_name = special_imports.get(library) or library.replace("-", "_")
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False


def _extract_structured_output_schema(response_format) -> Optional[str]:
    """
    Extract JSON schema string from a response_format value.
    Handles dict (json_schema type), Pydantic models, and TypeAdapters.
    """
    try:
        import json as _json

        if isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                schema = (response_format.get("json_schema") or {}).get("schema")
                if schema:
                    return _json.dumps(schema)
            return None

        # Pydantic BaseModel subclass or instance
        if hasattr(response_format, "model_json_schema") and callable(
            response_format.model_json_schema
        ):
            return _json.dumps(response_format.model_json_schema())

        # TypeAdapter or other pydantic construct
        try:
            import pydantic

            return _json.dumps(pydantic.TypeAdapter(response_format).json_schema())
        except Exception:
            pass

        # Last resort: try direct JSON dump
        try:
            return _json.dumps(response_format)
        except Exception:
            return None

    except Exception:
        return None
