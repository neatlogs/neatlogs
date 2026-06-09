"""
LangChain Python example with Neatlogs.

``instrumentations=["langchain"]`` captures LangChain LLM calls, chains, tools,
and retrievers automatically. The run self-roots — a WORKFLOW root is opened
automatically — so a bare ``llm.invoke()`` renders with no manual decorator.
(To group a multi-step run under one named root, add ``@neatlogs.span(kind="WORKFLOW")``.)

Points ChatOpenAI at OpenRouter (OpenAI-compatible) so it runs with just an
OpenRouter key.

Run:
    python examples/sdk_examples/langchain_basic.py

Env:
    OPENROUTER_API_KEY (required)
    LANGCHAIN_MODEL    (default: openai/gpt-4o-mini)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="langchain-basic-py",
    tags=["langchain", "python", "basic"],
    instrumentations=["langchain"],
)

# Import LangChain AFTER init() so it's patched.
from langchain_openai import ChatOpenAI  # noqa: E402


def main() -> None:
    model = os.getenv("LANGCHAIN_MODEL", "openai/gpt-4o-mini")
    llm = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        temperature=0.3,
        max_tokens=256,
    )

    # Bare invoke — self-roots into a WORKFLOW automatically.
    print(llm.invoke("In one sentence, what is LangChain?").content)

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
