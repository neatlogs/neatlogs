"""
DSPy Python example with Neatlogs.

Demonstrates the version-agnostic ``neatlogs.wrap()`` path for DSPy: wrapping a
``dspy.Module`` installs class-level hooks that trace module calls (CHAIN), LM
calls (LLM), and retrieval (RETRIEVER). The wrapped call self-roots — a WORKFLOW
root is opened automatically when there is no surrounding ``@neatlogs.span`` /
``trace()`` — so a single run renders in the dashboard with no extra code.

Two ways to instrument DSPy:
  1. ``neatlogs.wrap(module)``  — works on ANY dspy version (used here).
  2. ``neatlogs.init(instrumentations=["dspy"])`` — uses OpenInference, which
     requires ``dspy >= 2.6.0``. On older dspy it no-ops (emits no spans), so
     prefer ``wrap()`` if you can't upgrade.

Run:
    python examples/sdk_examples/dspy_basic.py

Env:
    OPENROUTER_API_KEY (required — DSPy's LM points at OpenRouter here)
    DSPY_MODEL         (default: openai/openai/gpt-4o-mini  — litellm-style slug)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import dspy
import neatlogs


class QA(dspy.Module):
    """A small two-step module: classify the question's topic, then answer it."""

    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.Predict("question -> topic")
        self.answer = dspy.Predict("question, topic -> answer")

    def forward(self, question: str):
        topic = self.classify(question=question).topic
        answer = self.answer(question=question, topic=topic).answer
        return dspy.Prediction(topic=topic, answer=answer)


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="dspy-basic-py",
        tags=["dspy", "python", "basic"],
    )

    # DSPy routes LM calls through litellm; point it at OpenRouter so this
    # example runs with just an OpenRouter key.
    model = os.getenv("DSPY_MODEL", "openai/openai/gpt-4o-mini")
    dspy.configure(
        lm=dspy.LM(
            model,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            max_tokens=256,
            temperature=0.3,
        )
    )

    # wrap() installs DSPy tracing hooks and self-roots, so this renders with no
    # manual trace() wrapper. The two predictor calls nest under the module run.
    qa = neatlogs.wrap(QA())

    result = qa(question="In one sentence, what is DSPy?")
    print("topic:", result.topic)
    print("answer:", result.answer)

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
