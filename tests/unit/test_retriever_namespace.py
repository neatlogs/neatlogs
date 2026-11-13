import json
from types import SimpleNamespace

import pytest

from neatlogs.decorators.orchestration import _retriever_postprocessor


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes = {}

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value


def test_retriever_postprocessor_emits_only_canonical_namespace() -> None:
    span = _RecordingSpan()

    _retriever_postprocessor(
        span,
        [
            "plain document",
            {
                "id": "doc-2",
                "content": "structured document",
                "score": 0.9,
                "metadata": {"source": "kb"},
            },
        ],
        {"query": "shipping policy"},
    )

    assert span.attributes == {
        "neatlogs.retriever.query": "shipping policy",
        "neatlogs.retriever.documents.0.content": "plain document",
        "neatlogs.retriever.documents.1.id": "doc-2",
        "neatlogs.retriever.documents.1.content": "structured document",
        "neatlogs.retriever.documents.1.score": 0.9,
        "neatlogs.retriever.documents.1.metadata": json.dumps({"source": "kb"}),
    }
    assert all(not key.startswith("neatlogs.retrieval.") for key in span.attributes)
    assert all(not key.startswith("retrieval.") for key in span.attributes)


def test_retriever_postprocessor_does_not_truncate_documents() -> None:
    span = _RecordingSpan()
    documents = [f"document-{index}-" + ("x" * 12_000) for index in range(25)]

    _retriever_postprocessor(span, documents, {"query": "q" * 12_000})

    assert span.attributes["neatlogs.retriever.query"] == "q" * 12_000
    assert span.attributes["neatlogs.retriever.documents.24.content"] == documents[24]
    assert len(span.attributes) == 26


@pytest.mark.parametrize("with_raw", [False, True])
def test_retriever_records_scored_document_objects(with_raw) -> None:
    span = _RecordingSpan()
    documents = [
        (SimpleNamespace(page_content="Returns within seven days", metadata={"source": "faq"}), 0.0)
    ]
    result = (documents, {"matches": []}) if with_raw else documents

    _retriever_postprocessor(span, result, {"query": "Returns?"})

    assert span.attributes["neatlogs.retriever.documents.0.content"] == "Returns within seven days"
    assert span.attributes["neatlogs.retriever.documents.0.score"] == 0.0
    assert json.loads(span.attributes["neatlogs.retriever.documents.0.metadata"]) == {
        "source": "faq"
    }
    assert "neatlogs.retriever.documents.1.content" not in span.attributes
