"""
Tools for the Marketing Strategy crew.

Provides a Gemini-grounded Google Search tool and a website analyser.
These give agents access to real-time web information with source citations.
"""

import os
import neatlogs
from crewai.tools import tool
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Gemini client  (lazy init — auto-instrumented via "google_genai")
# ---------------------------------------------------------------------------
_gemini_client = None
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_GROUNDED_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.2,
)


def _get_gemini_client() -> genai.Client:
    """Return a shared Gemini client, created on first use."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


@tool("Web Search Google")
@neatlogs.span(kind="TOOL", name="Web Search Google")
def search_web(query: str) -> str:
    """
    Search the internet using Google Search via Gemini grounding.

    Use this tool whenever you need up-to-date information about a company,
    market trends, competitor analysis, audience data, or any real-time facts.
    Provide a clear, specific search query.

    Returns a grounded answer with source citations.
    """
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=f"Search and provide a detailed factual summary for: {query}",
        config=_GROUNDED_CONFIG,
    )
    text = response.text or "No results found."

    # Append grounding sources when available
    sources = []
    try:
        metadata = response.candidates[0].grounding_metadata
        if metadata and metadata.grounding_chunks:
            for chunk in metadata.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    sources.append(f"- {chunk.web.title}: {chunk.web.uri}")
    except (AttributeError, IndexError):
        pass

    if sources:
        text += "\n\nSources:\n" + "\n".join(sources)

    return text


@tool("Analyze Website Content")
@neatlogs.span(kind="TOOL", name="Analyze Website Content")
def analyze_website(url: str) -> str:
    """
    Analyze and summarise the content of a specific website URL.

    Use this tool when you have a specific URL and need to understand what
    the company/product is about, their messaging, positioning, etc.
    """
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=(
            f"Visit and analyze this website: {url}\n\n"
            "Provide a detailed summary covering:\n"
            "1. What the company/product does\n"
            "2. Their main value propositions\n"
            "3. Target audience signals\n"
            "4. Key messaging and tone\n"
            "5. Notable features or differentiators"
        ),
        config=_GROUNDED_CONFIG,
    )
    return response.text or "Unable to analyze the website."
