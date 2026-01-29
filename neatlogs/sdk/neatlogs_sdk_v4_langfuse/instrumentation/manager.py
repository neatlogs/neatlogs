"""
Instrumentation manager for dual instrumentation (OpenInference + OpenLLMetry).
"""

import importlib
from typing import List, Set, Optional
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

from .registry import INSTRUMENTATION_REGISTRY, get_libraries_by_tag


class InstrumentationManager:
    """
    Manages dual instrumentation (OpenInference + OpenLLMetry).
    
    Features:
    - Tag-based instrumentation selection (e.g., "llm", "agent")
    - Explicit library selection (e.g., "openai", "langchain")
    - Always-on HTTP instrumentation for context propagation
    - Lazy loading (only instruments what's installed)
    - Dual convention support (best of both worlds)
    """
    
    def __init__(self, provider: TracerProvider, debug: bool = False, excluded_urls: Optional[str] = None):
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
        
        This is CRITICAL for maintaining span hierarchy across threads.
        Must be called before any other instrumentation.
        """
        try:
            ThreadingInstrumentor().instrument()
            if self.debug:
                print("✅ Instrumented threading (context propagation)")
        except Exception as e:
            if self.debug:
                print(f"⚠️  Failed to instrument threading: {e}")
    
    def instrument_http(self) -> None:
        """
        Always instrument HTTP libraries for context propagation.
        
        HTTP instrumentation ensures that HTTP calls made within
        LLM/TOOL/RETRIEVER spans are correctly parented.
        """
        http_libs = ["requests", "httpx", "urllib3", "aiohttp"]
        
        for lib in http_libs:
            # Check if library is installed
            if not self._is_library_installed(lib):
                if self.debug:
                    print(f"⏭️  Skipped HTTP: {lib} (not installed)")
                continue
            
            try:
                # Instrument with OpenLLMetry (has HTTP instrumentation)
                self._instrument_library(lib, convention="openllmetry")
                self.instrumented.add(lib)
                
                if self.debug:
                    print(f"✅ Instrumented HTTP: {lib}")
            except Exception as e:
                if self.debug:
                    print(f"⚠️  Failed to instrument {lib}: {e}")
    
    def instrument_mcp(self) -> None:
        """
        Instrument MCP for cross-process context propagation.
        
        This enables distributed tracing where:
        - MCP client (your agent) runs in process A
        - MCP server (tool provider) runs in process B (often stdio subprocess)
        - Traces are connected via W3C TraceContext in MCP protocol metadata
        
        Uses DUAL instrumentation:
        1. openinference-instrumentation-mcp: Context injection/extraction
        2. opentelemetry-instrumentation-mcp: Span creation for MCP operations
        """
        # Check if MCP is installed
        if not self._is_library_installed("mcp"):
            if self.debug:
                print("⏭️  Skipped MCP: not installed")
            return
        
        instrumented_any = False
        
        # 1. Context propagation layer (OpenInference)
        try:
            from openinference.instrumentation.mcp import MCPInstrumentor
            MCPInstrumentor().instrument(tracer_provider=self.provider)
            instrumented_any = True
            if self.debug:
                print("✅ MCP (OpenInference - context propagation)")
        except Exception as e:
            if self.debug:
                print(f"⚠️  MCP (OpenInference): {e}")
        
        # 2. Span creation layer (OpenLLMetry)
        try:
            from opentelemetry.instrumentation.mcp import McpInstrumentor
            McpInstrumentor().instrument(tracer_provider=self.provider)
            instrumented_any = True
            if self.debug:
                print("✅ MCP (OpenLLMetry - span creation)")
        except Exception as e:
            if self.debug:
                print(f"⚠️  MCP (OpenLLMetry): {e}")
        
        if instrumented_any:
            self.instrumented.add("mcp")
    
    def instrument(self, tags: Optional[List[str]] = None, libraries: Optional[List[str]] = None) -> None:
        """
        Instrument libraries based on tags and explicit library names.
        
        Args:
            tags: Semantic tags (e.g., ["llm", "agent", "retrieval"])
            libraries: Explicit library names (e.g., ["openai", "langchain"])
        """
        tags = tags or []
        libraries = libraries or []
        
        # Resolve tags to library names
        tag_libraries = set()
        for tag in tags:
            tag_libraries.update(get_libraries_by_tag(tag))
        
        # Combine with explicit libraries
        all_libraries = tag_libraries.union(set(libraries))
        
        # Instrument each library with DUAL convention
        for lib in all_libraries:
            if lib in self.instrumented:
                continue  # Skip already instrumented
            
            self._instrument_dual(lib)
    
    def _instrument_dual(self, library: str) -> None:
        """
        Instrument with BOTH OpenLLMetry and OpenInference (if available),
        or use custom neatlogs instrumentation if provided.

        Priority:
        1. Neatlogs (custom instrumentation) - if available, use ONLY this
        2. Dual (OpenLLMetry + OpenInference) - fallback to dual instrumentation

        This gives us:
        - Neatlogs: Custom unified instrumentation (replaces both)
        - OpenLLMetry: Streaming info, operational metrics
        - OpenInference: Cost tracking, span kinds, analytics attributes

        Args:
            library: Library name (e.g., "openai", "langchain")
        """
        # Check if library is installed
        if not self._is_library_installed(library):
            if self.debug:
                print(f"⏭️  Skipped: {library} (not installed)")
            return

        # Get instrumentation info from registry
        info = INSTRUMENTATION_REGISTRY["libraries"].get(library)
        if not info:
            if self.debug:
                print(f"⚠️  Unknown library: {library}")
            return

        instrumented_any = False

        # Check for custom neatlogs instrumentation first (takes priority)
        if info.get("neatlogs"):
            try:
                self._instrument_library(library, convention="neatlogs")
                instrumented_any = True
                if self.debug:
                    print(f"✅ {library} (Neatlogs - custom unified)")

                # Mark as instrumented and return early (don't use dual)
                if instrumented_any:
                    self.instrumented.add(library)
                return
            except Exception as e:
                if self.debug:
                    print(f"⚠️  {library} (Neatlogs): {e}")
                # Fall through to dual instrumentation on error

        # Fallback to dual instrumentation (OpenLLMetry + OpenInference)

        # Instrument with OpenLLMetry (if available)
        if info.get("openllmetry"):
            try:
                self._instrument_library(library, convention="openllmetry")
                instrumented_any = True
                if self.debug:
                    print(f"✅ {library} (OpenLLMetry)")
            except Exception as e:
                if self.debug:
                    print(f"⚠️  {library} (OpenLLMetry): {e}")

        # Instrument with OpenInference (if available)
        if info.get("openinference"):
            try:
                self._instrument_library(library, convention="openinference")
                instrumented_any = True
                if self.debug:
                    print(f"✅ {library} (OpenInference)")
            except Exception as e:
                if self.debug:
                    print(f"⚠️  {library} (OpenInference): {e}")

        if instrumented_any:
            self.instrumented.add(library)
    
    def _instrument_library(self, library: str, convention: str) -> None:
        """
        Dynamically import and instrument a library.

        Args:
            library: Library name (e.g., "openai")
            convention: "openllmetry", "openinference", or "neatlogs"
        """
        info = INSTRUMENTATION_REGISTRY["libraries"][library]
        package_name = info.get(convention)

        if not package_name:
            return

        try:
            # Import the instrumentation package
            module = importlib.import_module(package_name)

            # Framework instrumentations (e.g. OpenLLMetry LangChain) often set
            # SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY=True to avoid downstream provider spans.
            # Neatlogs handles dedupe at export-time, so we patch provider instrumentations to
            # ignore that suppression. This lets us keep provider spans/events (e.g. streaming
            # chunk events) even when a framework instrumentation is enabled.
            if convention == "openllmetry" and library == "openai":
                self._patch_openllmetry_openai_ignore_language_model_suppression()

            # Get the instrumentor class name
            # Most follow pattern: {Library}Instrumentor
            # But there are many exceptions...
            instrumentor_class_name = self._get_instrumentor_class_name(library, convention)

            instrumentor_class = getattr(module, instrumentor_class_name)

            # For HTTP libraries, add excluded_urls to prevent export calls from being traced
            is_http_lib = library in ["requests", "httpx", "urllib3", "aiohttp"]
            if is_http_lib and self.excluded_urls:
                instrumentor_class().instrument(
                    tracer_provider=self.provider,
                    excluded_urls=self.excluded_urls
                )
            else:
                instrumentor_class().instrument(tracer_provider=self.provider)

        except Exception as e:
            raise Exception(f"Failed to instrument {library} with {convention}: {e}")

    def _patch_openllmetry_openai_ignore_language_model_suppression(self) -> None:
        """
        Monkeypatch OpenLLMetry OpenAI wrappers to ignore SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY.

        Why:
        - OpenLLMetry framework instrumentations (e.g. LangChain) set this flag to True and it
          prevents OpenLLMetry provider spans like `openai.chat` from being emitted.
        - We dedupe provider spans in Neatlogs (span_processor.py), but we still want:
          - provider span attributes (gen_ai.*)
          - streaming chunk events (llm.content.completion.chunk)

        This patch is local to the running process (no fork/patch of OpenLLMetry required).
        """
        try:
            from functools import wraps

            from opentelemetry import context as context_api
            from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
            from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

            # Import wrapper modules from the installed OpenLLMetry OpenAI package.
            from opentelemetry.instrumentation.openai.shared import (
                chat_wrappers,
                completion_wrappers,
                embeddings_wrappers,
                image_gen_wrappers,
            )

            if getattr(chat_wrappers, "_NEATLOGS_PATCHED_IGNORE_LM_SUPPRESS", False):
                return

            def _wrap_factory(factory_fn):
                """
                OpenLLMetry's wrappers are factories:
                  wrapper = chat_wrapper(tracer, ...)
                The suppression check happens inside the returned `wrapper`, so we must
                wrap the returned wrapper (not just the factory itself).
                """

                @wraps(factory_fn)
                def _patched_factory(*f_args, **f_kwargs):
                    inner = factory_fn(*f_args, **f_kwargs)

                    @wraps(inner)
                    def _patched_wrapper(wrapped, instance, args, kwargs):
                        # Still respect global instrumentation suppression to avoid recursion.
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
                completion_wrappers.completion_wrapper = _wrap_factory(completion_wrappers.completion_wrapper)
            if hasattr(completion_wrappers, "acompletion_wrapper"):
                completion_wrappers.acompletion_wrapper = _wrap_factory(completion_wrappers.acompletion_wrapper)

            # Embeddings
            if hasattr(embeddings_wrappers, "embeddings_wrapper"):
                embeddings_wrappers.embeddings_wrapper = _wrap_factory(embeddings_wrappers.embeddings_wrapper)
            if hasattr(embeddings_wrappers, "aembeddings_wrapper"):
                embeddings_wrappers.aembeddings_wrapper = _wrap_factory(embeddings_wrappers.aembeddings_wrapper)

            # Image generation
            # (OpenLLMetry's OpenAI image instrumentation is metric-only and naming differs)
            if hasattr(image_gen_wrappers, "image_gen_metrics_wrapper"):
                image_gen_wrappers.image_gen_metrics_wrapper = _wrap_factory(image_gen_wrappers.image_gen_metrics_wrapper)
            if hasattr(image_gen_wrappers, "aimage_gen_metrics_wrapper"):
                image_gen_wrappers.aimage_gen_metrics_wrapper = _wrap_factory(image_gen_wrappers.aimage_gen_metrics_wrapper)

            chat_wrappers._NEATLOGS_PATCHED_IGNORE_LM_SUPPRESS = True
            if self.debug:
                print("Patched OpenLLMetry OpenAI: ignore SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY")
        except Exception as e:
            # Best-effort: if patching fails, instrumentation should still function.
            if self.debug:
                print(f"⚠️  Failed to patch OpenLLMetry OpenAI suppression: {e}")
    
    def _get_instrumentor_class_name(self, library: str, convention: str) -> str:
        """
        Get the correct instrumentor class name for a library.

        Different libraries use different naming conventions.

        Args:
            library: Library name (e.g., "openai")
            convention: "openllmetry", "openinference", or "neatlogs"

        Returns:
            Instrumentor class name
        """
        # Neatlogs convention (custom instrumentations)
        if convention == "neatlogs":
            neatlogs_cases = {
                "openai": "OpenAIInstrumentor",
                # Add more custom instrumentations here as they're implemented
            }
            if library in neatlogs_cases:
                return neatlogs_cases[library]

        # Special cases that don't follow standard naming
        special_cases = {
            "openai": "OpenAIInstrumentor",
            "langchain": "LangChainInstrumentor" if convention == "openinference" else "LangchainInstrumentor",
            "urllib3": "URLLib3Instrumentor",
            "httpx": "HTTPXClientInstrumentor",
            "aiohttp": "AioHttpClientInstrumentor",
            "llamaindex": "LlamaIndexInstrumentor" if convention == "openinference" else "LlamaindexInstrumentor",
            "google_generativeai": "GoogleGenerativeAIInstrumentor",
            "google_genai": "GoogleGenAIInstrumentor",
            "google_adk": "GoogleADKInstrumentor",
            "huggingface_hub": "HuggingfaceHubInstrumentor",
            "alephalpha": "AlephAlphaInstrumentor",
            "mistralai": "MistralAIInstrumentor",
            "vertexai": "VertexAIInstrumentor",
            "litellm": "LiteLLMInstrumentor",
            "crewai": "CrewAIInstrumentor",
            "dspy": "DSPyInstrumentor",
            "chromadb": "ChromaInstrumentor",
            "beeai": "BeeAIInstrumentor",
            "openai_agents": "OpenAIAgentsInstrumentor",
            "pydantic_ai": "PydanticAIInstrumentor",
            "mcp": "MCPInstrumentor" if convention == "openinference" else "McpInstrumentor",  # OpenInference=MCP, OpenLLMetry=Mcp
        }

        if library in special_cases:
            return special_cases[library]

        # Default: capitalize first letter
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
            # Convert library name to import name (handle special cases).
            #
            # NOTE: Some registry keys are "logical" names (e.g. google_genai) whose
            # import path is different (google.genai). If we don't map these, we will
            # incorrectly skip instrumentation with "not installed".
            special_imports = {
                # New Google GenAI SDK (pip: google-genai)
                "google_genai": "google.genai",
                # Older Google Generative AI SDK (pip: google-generativeai)
                "google_generativeai": "google.generativeai",
                # Milvus python client (pip: pymilvus)
                "milvus": "pymilvus",
            }
            import_name = special_imports.get(library) or library.replace("-", "_")
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
