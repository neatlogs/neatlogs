"""
Simple in-process tools for the LangGraph multi-agent research workflow.
No external HTTP calls — instant responses for reliable tracing demos.
"""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current news and information on a topic."""
    return (
        f"Web search results for '{query}':\n"
        f"- Recent developments show significant progress in this area.\n"
        f"- Industry experts highlight growing investment and adoption.\n"
        f"- Key players are actively publishing findings and case studies.\n"
        f"- Multiple startups and research groups are advancing the field."
    )


@tool
def wiki_search(query: str) -> str:
    """Search for encyclopedic background and foundational definitions on a topic."""
    return (
        f"Wikipedia summary for '{query}':\n"
        f"- The field originated in the mid-20th century with foundational theoretical work.\n"
        f"- Core principles include superposition, entanglement, and interference.\n"
        f"- Applications span medicine, finance, logistics, and materials science.\n"
        f"- Leading research institutions include MIT, Google, IBM, and national labs."
    )


@tool
def arxiv_search(query: str) -> str:
    """Search for recent academic papers and research findings on a topic."""
    return (
        f"ArXiv papers for '{query}':\n"
        f"- [2024] 'Advances in {query}: A Systematic Review' — 94% accuracy improvement.\n"
        f"- [2024] 'Benchmarking Methods for {query}' — new state-of-the-art baselines.\n"
        f"- [2025] 'Scaling {query} to Production' — practical deployment framework.\n"
        f"- [2025] 'Hybrid Approaches in {query}' — combines classical and quantum methods."
    )
