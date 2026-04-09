"""
Tools for the Marketing Strategy crew.

Provides a Gemini-grounded Google Search tool and a website analyser.
These give agents access to real-time web information with source citations.
"""

import os
from crewai.tools import BaseTool
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator


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
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _coerce_str(v: object) -> str:
    """Coerce weak-model structured inputs like {'description': '...', 'type': 'text'} to str."""
    if isinstance(v, dict):
        return v.get("description") or v.get("text") or v.get("query") or v.get("url") or str(v)
    return str(v)


class _SearchArgs(BaseModel):
    query: str = Field(..., description="Plain text search query, e.g. 'CrewAI competitors 2025'")

    @field_validator("query", mode="before")
    @classmethod
    def coerce(cls, v: object) -> str:
        return _coerce_str(v)


class _WebsiteArgs(BaseModel):
    url: str = Field(..., description="Full URL to analyze, e.g. 'https://crewai.com'")

    @field_validator("url", mode="before")
    @classmethod
    def coerce(cls, v: object) -> str:
        return _coerce_str(v)


class SearchWebTool(BaseTool):
    name: str = "Web Search Google"
    description: str = (
        "Search the internet using Google Search via Gemini grounding. "
        "Use this tool whenever you need up-to-date information about a company, "
        "market trends, competitor analysis, audience data, or any real-time facts. "
        "Provide a clear, specific search query as a plain string. "
        "Returns a grounded answer with source citations."
    )
    args_schema: type[BaseModel] = _SearchArgs

    def _run(self, query: str) -> str:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=f"Search and provide a detailed factual summary for: {query}",
            config=_GROUNDED_CONFIG,
        )
        text = response.text or "No results found."

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


class AnalyzeWebsiteTool(BaseTool):
    name: str = "Analyze Website Content"
    description: str = (
        "Analyze and summarise the content of a specific website URL. "
        "Use this tool when you have a specific URL and need to understand what "
        "the company/product is about, their messaging, positioning, etc. "
        "Provide the URL as a plain string."
    )
    args_schema: type[BaseModel] = _WebsiteArgs

    def _run(self, url: str) -> str:
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


search_web = SearchWebTool()
analyze_website = AnalyzeWebsiteTool()
