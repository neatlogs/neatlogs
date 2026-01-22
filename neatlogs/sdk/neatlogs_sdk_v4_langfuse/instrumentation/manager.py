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
        Instrument with BOTH OpenLLMetry and OpenInference (if available).
        
        This gives us:
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
            convention: "openllmetry" or "openinference"
        """
        info = INSTRUMENTATION_REGISTRY["libraries"][library]
        package_name = info[convention]
        
        if not package_name:
            return
        
        try:
            # Import the instrumentation package
            module = importlib.import_module(package_name)
            
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
    
    def _get_instrumentor_class_name(self, library: str, convention: str) -> str:
        """
        Get the correct instrumentor class name for a library.
        
        Different libraries use different naming conventions.
        """
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
            "mcp": "MCPInstrumentor",
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
            # Convert library name to import name (handle special cases)
            import_name = library.replace('-', '_')
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
