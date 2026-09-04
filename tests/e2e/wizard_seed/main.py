def retrieve_context(query: str) -> list[str]:
    return [f"documentation for {query}"]


def answer_question(query: str, documents: list[str]) -> str:
    return f"{query}: {documents[0]}"


def run_support_agent(query: str) -> str:
    documents = retrieve_context(query)
    return answer_question(query, documents)


if __name__ == "__main__":
    print(run_support_agent("launch readiness"))
