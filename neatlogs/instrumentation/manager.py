"""
Instrumentation manager.
"""

import importlib
import logging
from typing import List, Optional, Set

from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from .registry import INSTRUMENTATION_REGISTRY, get_libraries_by_tag
from .http_context_propagation import patch_http_context_propagation

logger = logging.getLogger(__name__)


class InstrumentationManager:

    def __init__(
        self, provider: TracerProvider, debug: bool = False, excluded_urls: Optional[str] = None
    ):
        """
        Initialize the instrumentation manager.

        Args:
            provider: OpenTelemetry tracer provider
            debug: Enable debug logging
            excluded_urls: Comma-separated URLs to exclude from HTTP tracing
        """
        self.provider = provider
        self.debug = debug
        self.excluded_urls = excluded_urls
        self.instrumented: Set[str] = set()

    def instrument_threading(self) -> None:
        """
        Instrument threading for context propagation.
        """
        try:
            ThreadingInstrumentor().instrument()
            if self.debug:
                logger.info("✅ Instrumented threading (context propagation)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to instrument threading: {e}")

    def instrument_http(self) -> None:
        """
        Always instrument HTTP libraries for context propagation.
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

        # Server-side frameworks are not always available as OTel dependencies in user envs.
        # Patch context propagation for common libraries/frameworks so cross-service calls
        # can still share a trace_id even without full HTTP auto-instrumentation packages.
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
        """
        if not self._is_library_installed("mcp"):
            if self.debug:
                logger.info("⏭️  Skipped MCP: not installed")
            return

        instrumented_any = False

        try:
            from openinference.instrumentation.mcp import MCPInstrumentor

            MCPInstrumentor().instrument(tracer_provider=self.provider)
            instrumented_any = True
            if self.debug:
                logger.info("✅ MCP (OpenInference - context propagation)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  MCP (OpenInference): {e}")

        try:
            from opentelemetry.instrumentation.mcp import McpInstrumentor

            McpInstrumentor().instrument(tracer_provider=self.provider)
            instrumented_any = True
            if self.debug:
                logger.info("✅ MCP (OpenLLMetry - span creation)")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  MCP (OpenLLMetry): {e}")

        if instrumented_any:
            self.instrumented.add("mcp")

    def instrument(
        self, tags: Optional[List[str]] = None, libraries: Optional[List[str]] = None
    ) -> None:
        """
        Instrument libraries based on tags and explicit library names.
        """
        tags = tags or []
        libraries = libraries or []
        tag_libraries = set()
        for tag in tags:
            tag_libraries.update(get_libraries_by_tag(tag))
        all_libraries = tag_libraries.union(set(libraries))
        for lib in all_libraries:
            if lib in self.instrumented:
                continue

            self._instrument_dual(lib)

    def _instrument_dual(self, library: str) -> None:
        if not self._is_library_installed(library):
            if self.debug:
                logger.info(f"⏭️  Skipped: {library} (not installed)")
            return

        info = INSTRUMENTATION_REGISTRY["libraries"].get(library)
        if not info:
            if self.debug:
                logger.warning(f"⚠️  Unknown library: {library}")
            return

        instrumented_any = False

        if info.get("neatlogs"):
            try:
                self._instrument_library(library, convention="neatlogs")
                instrumented_any = True
                if self.debug:
                    logger.info(f"✅ {library} (Neatlogs - custom unified)")

                if instrumented_any:
                    self.instrumented.add(library)
                return
            except Exception as e:
                if self.debug:
                    logger.warning(f"⚠️  {library} (Neatlogs): {e}")

        if info.get("openllmetry"):
            try:
                self._instrument_library(library, convention="openllmetry")
                instrumented_any = True
                if self.debug:
                    logger.info(f"✅ {library} (OpenLLMetry)")
            except Exception as e:
                if self.debug:
                    logger.warning(f"⚠️  {library} (OpenLLMetry): {e}")

        if info.get("openinference"):
            try:
                self._instrument_library(library, convention="openinference")
                instrumented_any = True
                if self.debug:
                    logger.info(f"✅ {library} (OpenInference)")
            except Exception as e:
                if self.debug:
                    logger.warning(f"⚠️  {library} (OpenInference): {e}")

        if instrumented_any:
            self.instrumented.add(library)

    def _instrument_library(self, library: str, convention: str) -> None:
        """
        Dynamically import and instrument a library.
        """
        info = INSTRUMENTATION_REGISTRY["libraries"][library]
        package_name = info.get(convention)

        if not package_name:
            return

        try:
            module = importlib.import_module(package_name)
            if convention == "openllmetry" and library == "openai":
                self._patch_openllmetry_openai_ignore_language_model_suppression()
            if convention == "openllmetry" and library == "langchain":
                self._patch_openllmetry_langchain_callback_handler_errors()
            if convention == "openllmetry" and library == "cohere":
                self._patch_openllmetry_cohere_safe_token_sums()
            instrumentor_class_name = self._get_instrumentor_class_name(library, convention)

            instrumentor_class = getattr(module, instrumentor_class_name)

            is_http_lib = library in ["requests", "httpx", "urllib3", "aiohttp"]
            if is_http_lib and self.excluded_urls:
                instrumentor_class().instrument(
                    tracer_provider=self.provider, excluded_urls=self.excluded_urls
                )
            else:
                instrumentor_class().instrument(tracer_provider=self.provider)

            if convention == "openinference" and library == "litellm":
                self._patch_openinference_litellm_ignore_instrumentation_suppression()

        except Exception as e:
            raise Exception(f"Failed to instrument {library} with {convention}: {e}")

    def _patch_openllmetry_cohere_safe_token_sums(self) -> None:
        """
        Patch OpenLLMetry Cohere instrumentation to handle token counts that are present
        but set to null/None (observed in some responses, e.g. rerank).

        Without this, OpenLLMetry can log:
          TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'
        and usage attributes won't be set for that span.
        """
        try:
            import opentelemetry.instrumentation.cohere as cohere_inst
            import opentelemetry.instrumentation.cohere.span_utils as span_utils
            from opentelemetry.semconv._incubating.attributes import (
                gen_ai_attributes as GenAIAttributes,
            )
            from opentelemetry.semconv_ai import SpanAttributes

            if getattr(cohere_inst, "_NEATLOGS_PATCHED_SAFE_TOKEN_SUMS", False):
                return

            def _safe_int(value: object, default: int = 0) -> int:
                if value is None:
                    return default
                try:
                    return int(value)  # type: ignore[arg-type]
                except Exception:
                    return default

            def _patched_set_span_response_attributes(span, response):
                if not span.is_recording():
                    return

                response_dict = span_utils.to_dict(response)

                # Cohere API v1
                if response_dict.get("response_id"):
                    span_utils._set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_RESPONSE_ID,
                        response_dict.get("response_id"),
                    )
                # Cohere API v2
                elif response_dict.get("id"):
                    span_utils._set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_RESPONSE_ID,
                        response_dict.get("id"),
                    )

                # Cohere v4
                token_count = response_dict.get("token_count")
                if token_count:
                    token_count_dict = span_utils.to_dict(token_count)
                    input_tokens = _safe_int(token_count_dict.get("prompt_tokens"))
                    output_tokens = _safe_int(token_count_dict.get("response_tokens"))
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                        input_tokens + output_tokens,
                    )
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                        output_tokens,
                    )
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                        input_tokens,
                    )

                # Cohere v5
                if response_dict.get("meta"):
                    meta_dict = span_utils.to_dict(response_dict.get("meta", {}))
                    billed_units_dict = span_utils.to_dict(meta_dict.get("billed_units", {}))
                    input_tokens = _safe_int(billed_units_dict.get("input_tokens"))
                    output_tokens = _safe_int(billed_units_dict.get("output_tokens"))

                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                        input_tokens + output_tokens,
                    )
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                        output_tokens,
                    )
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                        input_tokens,
                    )

                # Cohere API v2
                if response_dict.get("usage"):
                    usage_dict = span_utils.to_dict(response_dict.get("usage", {}))
                    billed_units_dict = span_utils.to_dict(usage_dict.get("billed_units", {}))
                    input_tokens = _safe_int(billed_units_dict.get("input_tokens"))
                    output_tokens = _safe_int(billed_units_dict.get("output_tokens"))
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                        input_tokens + output_tokens,
                    )
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                        output_tokens,
                    )
                    span_utils._set_span_attribute(
                        span,
                        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                        input_tokens,
                    )

            wrapped = span_utils.dont_throw(_patched_set_span_response_attributes)
            span_utils.set_span_response_attributes = wrapped
            # The instrumentation module imports this into its namespace at import-time.
            cohere_inst.set_span_response_attributes = wrapped

            cohere_inst._NEATLOGS_PATCHED_SAFE_TOKEN_SUMS = True
            if self.debug:
                logger.debug("Patched OpenLLMetry Cohere: safe token sums")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OpenLLMetry Cohere token sums: {e}")

    def _patch_openllmetry_langchain_callback_handler_errors(self) -> None:
        """
        Patch OpenLLMetry LangChain callback handler to:
        - Avoid KeyError when error callbacks fire after the span was already removed
        - Use correct `Span.set_status(...)` call shape (prevents OTel warnings)
        """
        try:
            import opentelemetry.instrumentation.langchain.callback_handler as cb
            from opentelemetry.trace.status import Status, StatusCode

            if getattr(cb, "_NEATLOGS_PATCHED_HANDLE_ERROR_GUARDS", False):
                return

            original_handle_error = cb.TraceloopCallbackHandler._handle_error

            def _patched_handle_error(
                self,
                error,
                run_id,
                parent_run_id=None,
                **kwargs,
            ):
                # If this run_id was never started or already ended, skip emitting error to avoid KeyError noise.
                holder = getattr(self, "spans", {}).get(run_id)
                if holder is None:
                    return

                span = holder.span
                try:
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                except Exception:
                    # Never let tracing errors break user code paths
                    return original_handle_error(self, error, run_id, parent_run_id, **kwargs)
                try:
                    span.record_exception(error)
                except Exception:
                    pass
                try:
                    self._end_span(span, run_id)
                except Exception:
                    pass

            cb.TraceloopCallbackHandler._handle_error = _patched_handle_error
            cb._NEATLOGS_PATCHED_HANDLE_ERROR_GUARDS = True
        except Exception as e:
            if self.debug:
                logger.warning(
                    f"⚠️  Failed to patch OpenLLMetry LangChain callback handler: {e}"
                )

    def _patch_openinference_litellm_ignore_instrumentation_suppression(self) -> None:
        try:
            from functools import wraps

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
                logger.debug("Patched OpenInference LiteLLM: ignore _SUPPRESS_INSTRUMENTATION_KEY")
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OpenInference LiteLLM suppression: {e}")

    def _patch_openllmetry_openai_ignore_language_model_suppression(self) -> None:
        try:
            from functools import wraps

            from opentelemetry import context as context_api
            from opentelemetry.instrumentation.openai.shared import (
                chat_wrappers,
                completion_wrappers,
                embeddings_wrappers,
                image_gen_wrappers,
            )
            from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
            from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

            if getattr(chat_wrappers, "_NEATLOGS_PATCHED_IGNORE_LM_SUPPRESS", False):
                return

            def _wrap_factory(factory_fn):

                @wraps(factory_fn)
                def _patched_factory(*f_args, **f_kwargs):
                    inner = factory_fn(*f_args, **f_kwargs)

                    @wraps(inner)
                    def _patched_wrapper(wrapped, instance, args, kwargs):
                        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                            return inner(wrapped, instance, args, kwargs)

                        token = None
                        try:
                            token = context_api.attach(
                                context_api.set_value(
                                    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, False
                                )
                            )
                        except Exception:
                            token = None

                        try:
                            return inner(wrapped, instance, args, kwargs)
                        finally:
                            if token is not None:
                                try:
                                    context_api.detach(token)
                                except Exception:
                                    pass

                    return _patched_wrapper

                return _patched_factory

            # Chat
            if hasattr(chat_wrappers, "chat_wrapper"):
                chat_wrappers.chat_wrapper = _wrap_factory(chat_wrappers.chat_wrapper)
            if hasattr(chat_wrappers, "achat_wrapper"):
                chat_wrappers.achat_wrapper = _wrap_factory(chat_wrappers.achat_wrapper)

            # Completions
            if hasattr(completion_wrappers, "completion_wrapper"):
                completion_wrappers.completion_wrapper = _wrap_factory(
                    completion_wrappers.completion_wrapper
                )
            if hasattr(completion_wrappers, "acompletion_wrapper"):
                completion_wrappers.acompletion_wrapper = _wrap_factory(
                    completion_wrappers.acompletion_wrapper
                )

            # Embeddings
            if hasattr(embeddings_wrappers, "embeddings_wrapper"):
                embeddings_wrappers.embeddings_wrapper = _wrap_factory(
                    embeddings_wrappers.embeddings_wrapper
                )
            if hasattr(embeddings_wrappers, "aembeddings_wrapper"):
                embeddings_wrappers.aembeddings_wrapper = _wrap_factory(
                    embeddings_wrappers.aembeddings_wrapper
                )

            if hasattr(image_gen_wrappers, "image_gen_metrics_wrapper"):
                image_gen_wrappers.image_gen_metrics_wrapper = _wrap_factory(
                    image_gen_wrappers.image_gen_metrics_wrapper
                )
            if hasattr(image_gen_wrappers, "aimage_gen_metrics_wrapper"):
                image_gen_wrappers.aimage_gen_metrics_wrapper = _wrap_factory(
                    image_gen_wrappers.aimage_gen_metrics_wrapper
                )

            chat_wrappers._NEATLOGS_PATCHED_IGNORE_LM_SUPPRESS = True
            if self.debug:
                logger.debug(
                    "Patched OpenLLMetry OpenAI: ignore SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY"
                )
        except Exception as e:
            if self.debug:
                logger.warning(f"⚠️  Failed to patch OpenLLMetry OpenAI suppression: {e}")

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
            "llamaindex": (
                "LlamaIndexInstrumentor"
                if convention == "openinference"
                else "LlamaindexInstrumentor"
            ),
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
            "mcp": "MCPInstrumentor" if convention == "openinference" else "McpInstrumentor",
        }

        if library in special_cases:
            return special_cases[library]

        return f"{library.capitalize()}Instrumentor"

    def _is_library_installed(self, library: str) -> bool:
        """
        Check if a library is installed.

        Args:
            library: Library name

        Returns:
            True if library is installed
        """
        try:
            special_imports = {
                "google_genai": "google.genai",
                # Older Google Generative AI SDK (pip: google-generativeai)
                "google_generativeai": "google.generativeai",
                # Azure AI Inference SDK (pip: azure-ai-inference)
                "azure_ai_inference": "azure.ai.inference",
                # AWS Bedrock uses boto3 (pip: boto3)
                "bedrock": "boto3",
                # Milvus python client (pip: pymilvus)
                "milvus": "pymilvus",
                # Qdrant python client (pip: qdrant-client)
                "qdrant": "qdrant_client",
            }
            import_name = special_imports.get(library) or library.replace("-", "_")
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
