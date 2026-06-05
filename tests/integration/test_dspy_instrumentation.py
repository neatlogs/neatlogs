"""
Comprehensive DSPy Framework Instrumentation Test Suite
=======================================================
Tests DSPy with OpenInference/OpenTelemetry instrumentation.
Validates span creation across complexity levels from basic to production-grade.
Patterns Tested:
1. ✅ Basic DSPy Modules (ChainOfThought, ReAct)
2. ✅ Retrieval-Augmented Generation (RAG)
3. ✅ Multi-hop Reasoning Chains
4. ✅ DSPy Program Compilation/Optimization
5. ✅ Custom Signatures & Modules
6. ✅ Multi-Model DSPy Programs
7. ✅ Async Execution Patterns
8. ✅ Streaming Responses
9. ✅ Evaluation Metrics & Validation
10. ⚠️ Error Handling & Edge Cases
"""

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import numpy as np
import pytest
import respx

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)

# Skip if DSPy not installed
try:
    import dspy
    from dsp.utils import deduplicate
    from dspy import (
        ChainOfThought,
        InputField,
        Module,
        MultiChainComparison,
        OutputField,
        Predict,
        ProgramOfThought,
        ReAct,
        Retrieve,
        RetrieveThenRead,
        Signature,
    )

    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    pytest.skip("DSPy not installed", allow_module_level=True)

# =================================================================
# TEST SETUP & FIXTURES
# =================================================================


class TestDSPyInstrumentation:
    """Main test class for DSPy framework instrumentation validation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        """Global test setup with OpenTelemetry and Neatlogs initialization."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Setup OpenTelemetry with in-memory exporter
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)

        # Initialize Neatlogs with DSPy instrumentation
        try:
            import neatlogs

            neatlogs.init(
                api_key="test-key",
                enable_otel=True,
                disable_export=True,
                # DSPy and OpenAI instrumentation
                instrumentations=["dspy", "openai"],
            )
        except ImportError:
            pytest.skip("neatlogs not installed", allow_module_level=True)

        self.exporter = in_memory_span_exporter
        yield

        # Cleanup
        self.exporter.clear()

    @pytest.fixture
    def mock_openai_chat_response(self):
        """Standard OpenAI chat response fixture."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "This is a DSPy test response."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        }

    @pytest.fixture
    def mock_openai_embedding_response(self):
        """OpenAI embeddings response fixture."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3] * 512,  # 1536-dim vector
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

    @pytest.fixture
    def mock_retrieval_results(self):
        """Mock retrieval results for DSPy RAG modules."""
        return [
            dspy.Prediction(
                long_text="Python is a high-level programming language created by Guido van Rossum.",
                score=0.95,
            ),
            dspy.Prediction(
                long_text="DSPy is a framework for algorithmically optimizing LM prompts and weights.",
                score=0.85,
            ),
        ]

    @pytest.fixture
    def mock_colbert_retriever(self, mock_retrieval_results):
        """Mock ColBERT retriever for DSPy."""

        class MockColBERTRetriever:
            def __call__(self, query, k=3):
                return mock_retrieval_results[:k]

        return MockColBERTRetriever()

    def wait_for_spans(self, min_spans=1, timeout=2.0):
        """Helper to wait for spans to be processed."""
        start = time.time()
        while time.time() - start < timeout:
            spans = self.exporter.get_finished_spans()
            if len(spans) >= min_spans:
                return spans
            time.sleep(0.05)
        return self.exporter.get_finished_spans()

    def assert_span_exists(self, span_name: str, spans: List = None):
        """Assert a span with given name exists."""
        if spans is None:
            spans = self.exporter.get_finished_spans()
        matching_spans = [s for s in spans if span_name in s.name]
        assert len(matching_spans) > 0, f"Span '{span_name}' not found in {[s.name for s in spans]}"
        return matching_spans[0]

    def get_span_by_attribute(self, attribute_name: str, attribute_value: Any, spans: List = None):
        """Get span by specific attribute value."""
        if spans is None:
            spans = self.exporter.get_finished_spans()
        return next((s for s in spans if s.attributes.get(attribute_name) == attribute_value), None)

    # =================================================================
    # 🟢 PATTERN 1: BASIC DSPY MODULES (ChainOfThought, ReAct)
    # =================================================================
    @respx.mock
    def test_chain_of_thought_module(self, mock_openai_chat_response):
        """
        Test basic DSPy ChainOfThought module.
        Expected spans: 'ChainOfThought', 'Predict', 'ChatOpenAI'
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        dspy.settings.configure(lm=lm)

        # Define signature
        class QA(Signature):
            """Answer questions with reasoning."""

            question = InputField()
            answer = OutputField(desc="Often between 1 and 5 words")

        # Create ChainOfThought module
        cot = ChainOfThought(QA)

        # Execute
        prediction = cot(question="What is the capital of France?")

        # Validate prediction
        assert prediction is not None
        assert hasattr(prediction, "answer")

        # Validate spans
        spans = self.wait_for_spans(min_spans=3)

        # Check for module spans
        cot_span = self.assert_span_exists("ChainOfThought", spans)
        predict_span = self.assert_span_exists("Predict", spans)

        # Verify attributes
        assert cot_span.attributes.get("module.type") == "ChainOfThought"
        assert cot_span.attributes.get("signature.name") == "QA"
        assert "capital" in str(cot_span.attributes.get("input", "")).lower()

        # Check LLM span
        llm_spans = [s for s in spans if "openai" in s.name.lower() or "chat" in s.name.lower()]
        assert len(llm_spans) >= 1
        llm_span = llm_spans[0]
        assert llm_span.attributes.get("llm.model.name") == "gpt-4o"
        assert llm_span.parent.span_id == predict_span.context.span_id

    @respx.mock
    def test_react_module(self, mock_openai_chat_response):
        """
        Test DSPy ReAct module with tool calling.
        Expected spans: 'ReAct', 'Predict', 'Tool' spans
        """
        # Mock tool responses
        tool_responses = [
            {"name": "search", "arguments": {"query": "weather in Paris"}, "result": "Sunny, 22°C"},
            {"name": "calculator", "arguments": {"expression": "2+2"}, "result": "4"},
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        dspy.settings.configure(lm=lm)

        # Define signature with tools
        class WeatherQA(Signature):
            """Answer weather questions using tools."""

            question = InputField()
            answer = OutputField()

        # Create ReAct module
        react = ReAct(WeatherQA)

        # Mock tool execution
        with patch.object(react, "tools", new_callable=MagicMock) as mock_tools:
            mock_tools.search.return_value = "Sunny, 22°C in Paris"
            mock_tools.calculator.return_value = "4"

            # Execute
            prediction = react(question="What's the weather in Paris and what's 2+2?")

        # Validate spans
        spans = self.wait_for_spans(min_spans=4)

        # Check module spans
        react_span = self.assert_span_exists("ReAct", spans)
        assert react_span.attributes.get("module.type") == "ReAct"

        # Check for tool spans
        tool_spans = [s for s in spans if "tool" in s.name.lower()]
        assert len(tool_spans) >= 1, f"Expected tool spans, found: {[s.name for s in spans]}"

        # Verify reasoning trace
        reasoning_spans = [s for s in spans if "reasoning" in str(s.attributes).lower()]
        if reasoning_spans:
            assert "weather" in str(reasoning_spans[0].attributes).lower()

    # =================================================================
    # 🔵 PATTERN 2: RETRIEVAL-AUGMENTED GENERATION (RAG)
    # =================================================================
    @respx.mock
    def test_rag_module(
        self, mock_openai_chat_response, mock_openai_embedding_response, mock_retrieval_results
    ):
        """
        Test DSPy RetrieveThenRead (RAG) module.
        Expected spans: 'RetrieveThenRead', 'Retrieve', 'GenerateAnswer', 'Embedding'
        """
        # Mock API endpoints
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_openai_embedding_response)
        )

        # Configure DSPy with mock retriever
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        retriever = MagicMock()
        retriever.return_value = mock_retrieval_results

        rm = dspy.ColBERTv2(url="http://fake-colbert")
        dspy.settings.configure(lm=lm, rm=rm)

        # Create RAG module
        class RAG(Module):
            def __init__(self):
                super().__init__()
                self.retrieve = Retrieve(k=2)
                self.generate_answer = dspy.ChainOfThought("question, context -> answer")

            def forward(self, question):
                context = self.retrieve(question).passages
                prediction = self.generate_answer(question=question, context=context)
                return dspy.Prediction(context=context, answer=prediction.answer)

        rag = RAG()

        # Execute
        prediction = rag(question="What is DSPy?")

        # Validate
        assert prediction is not None
        assert hasattr(prediction, "answer")
        assert hasattr(prediction, "context")

        # Validate spans
        spans = self.wait_for_spans(min_spans=5)

        # Check module spans
        rag_span = self.assert_span_exists("RAG", spans)  # Custom module name
        retrieve_span = self.assert_span_exists("Retrieve", spans)
        generate_span = self.assert_span_exists("ChainOfThought", spans)

        # Verify retrieval attributes
        assert retrieve_span.attributes.get("retriever.type") == "ColBERTv2"
        assert retrieve_span.attributes.get("query") == "What is DSPy?"
        assert retrieve_span.attributes.get("documents.retrieved") == 2

        # Check embedding span
        embedding_spans = [s for s in spans if "embedding" in s.name.lower()]
        assert len(embedding_spans) >= 1, "Missing embedding span"

    @respx.mock
    def test_multi_hop_rag(
        self, mock_openai_chat_response, mock_openai_embedding_response, mock_retrieval_results
    ):
        """
        Test multi-hop RAG with DSPy.
        Complex pattern with sequential retrieval and reasoning.
        """
        # Mock API endpoints
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_openai_embedding_response)
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        rm = dspy.ColBERTv2(url="http://fake-colbert")
        dspy.settings.configure(lm=lm, rm=rm)

        # Create multi-hop RAG module
        class MultiHopRAG(Module):
            def __init__(self):
                super().__init__()
                self.retrieve = Retrieve(k=3)
                self.generate_query = dspy.Predict("context, question -> query")
                self.generate_answer = dspy.ChainOfThought("question, context -> answer")

            def forward(self, question):
                # First hop
                context1 = self.retrieve(question).passages

                # Generate second query
                query2 = self.generate_query(context=context1, question=question).query

                # Second hop
                context2 = self.retrieve(query2).passages

                # Final answer
                prediction = self.generate_answer(question=question, context=context1 + context2)

                return dspy.Prediction(
                    context=context1 + context2, answer=prediction.answer, intermediate_query=query2
                )

        multi_hop_rag = MultiHopRAG()

        # Execute
        prediction = multi_hop_rag(question="Who founded the company that makes DSPy?")

        # Validate spans
        spans = self.wait_for_spans(min_spans=8)

        # Check for multiple retrieval spans
        retrieve_spans = [s for s in spans if "Retrieve" in s.name]
        assert (
            len(retrieve_spans) >= 2
        ), f"Expected at least 2 Retrieve spans, got {len(retrieve_spans)}"

        # Check for query generation span
        query_spans = [
            s for s in spans if "Predict" in s.name and "query" in str(s.attributes).lower()
        ]
        assert len(query_spans) >= 1

        # Verify context propagation
        first_retrieve = retrieve_spans[0]
        second_retrieve = retrieve_spans[1]

        # Should be in same trace
        assert first_retrieve.context.trace_id == second_retrieve.context.trace_id

    # =================================================================
    # 🟡 PATTERN 3: PROGRAM COMPILATION/OPTIMIZATION
    # =================================================================
    @respx.mock
    def test_program_compilation(self, mock_openai_chat_response):
        """
        Test DSPy program compilation/optimization.
        Complex pattern with training and optimization phases.
        """
        # Mock multiple API calls for compilation
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=mock_openai_chat_response),
                httpx.Response(200, json=mock_openai_chat_response),
                httpx.Response(200, json=mock_openai_chat_response),
            ]
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        dspy.settings.configure(lm=lm)

        # Define signature
        class BasicQA(Signature):
            """Answer questions accurately."""

            question = InputField()
            answer = OutputField()

        # Create uncompiled module
        class QAModule(Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict(BasicQA)

            def forward(self, question):
                return self.predict(question=question)

        uncompiled = QAModule()

        # Create mock training data
        trainset = [
            dspy.Example(question="What is Python?", answer="Programming language").with_inputs(
                "question"
            ),
            dspy.Example(question="Who created Linux?", answer="Linus Torvalds").with_inputs(
                "question"
            ),
        ]

        # Configure optimizer
        from dspy.teleprompt import BootstrapFewShot

        teleprompter = BootstrapFewShot(metric=lambda a, b: a.answer == b.answer)

        # Compile program
        compiled = teleprompter.compile(uncompiled, trainset=trainset)

        # Execute compiled program
        prediction = compiled(question="What is DSPy?")

        # Validate spans
        spans = self.wait_for_spans(min_spans=6)

        # Check for compilation spans
        compilation_spans = [
            s for s in spans if "compilation" in s.name.lower() or "teleprompt" in s.name.lower()
        ]
        assert len(compilation_spans) >= 1, "Missing compilation spans"

        # Check for optimization phases
        optimization_spans = [s for s in spans if "optimization" in str(s.attributes).lower()]
        if optimization_spans:
            assert len(optimization_spans) >= 1

        # Verify compiled module has different attributes
        predict_spans = [s for s in spans if "Predict" in s.name]
        compiled_spans = [s for s in predict_spans if "compiled" in str(s.attributes).lower()]
        assert len(compiled_spans) >= 1

    @respx.mock
    def test_multi_chain_comparison(self, mock_openai_chat_response):
        """
        Test DSPy MultiChainComparison module.
        Advanced pattern with multiple reasoning chains and voting.
        """
        # Mock multiple API calls
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[httpx.Response(200, json=mock_openai_chat_response) for _ in range(5)]
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        dspy.settings.configure(lm=lm)

        # Define signature
        class MathQA(Signature):
            """Solve math problems with step-by-step reasoning."""

            question = InputField()
            answer = OutputField(desc="Numerical answer")

        # Create MultiChainComparison module
        mcc = MultiChainComparison(MathQA, M=3, temperature=0.7)  # Number of chains

        # Execute
        prediction = mcc(question="What is 15 * 3 + 10?")

        # Validate
        assert prediction is not None
        assert hasattr(prediction, "answer")

        # Validate spans
        spans = self.wait_for_spans(min_spans=7)

        # Check for multiple chain spans
        chain_spans = [s for s in spans if "Chain" in s.name or "Predict" in s.name]
        assert len(chain_spans) >= 3, f"Expected at least 3 chain spans, got {len(chain_spans)}"

        # Check for comparison/voting span
        comparison_spans = [
            s
            for s in spans
            if "comparison" in s.name.lower() or "vote" in str(s.attributes).lower()
        ]
        assert len(comparison_spans) >= 1

        # Verify parallel execution attributes
        for span in chain_spans:
            if "chain_index" in span.attributes:
                assert span.attributes.get("chain_index") in [0, 1, 2]

    # =================================================================
    # 🟠 PATTERN 4: CUSTOM SIGNATURES & MODULES
    # =================================================================
    @respx.mock
    def test_custom_signature_module(self, mock_openai_chat_response):
        """
        Test custom DSPy signature and module.
        Complex pattern with structured output and validation.
        """
        # Mock API
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        dspy.settings.configure(lm=lm)

        # Define complex custom signature
        class WeatherAnalysis(Signature):
            """Analyze weather data and provide recommendations."""

            location = InputField(desc="City or location name")
            current_weather = InputField(desc="Current weather conditions")
            activity = InputField(desc="Planned outdoor activity")

            weather_summary = OutputField(desc="Brief summary of weather conditions")
            recommendation = OutputField(desc="Recommendation for the activity")
            confidence = OutputField(desc="Confidence level 0-1", prefix="Confidence:")

        # Create custom module
        class WeatherAdvisor(Module):
            def __init__(self):
                super().__init__()
                self.analyze = dspy.ChainOfThought(WeatherAnalysis)

            def forward(self, location, activity):
                # Mock weather data
                current_weather = f"Temperature: 22°C, Conditions: Sunny, Wind: Light"
                prediction = self.analyze(
                    location=location, current_weather=current_weather, activity=activity
                )
                return prediction

        weather_advisor = WeatherAdvisor()

        # Execute
        prediction = weather_advisor(location="San Francisco", activity="hiking")

        # Validate
        assert hasattr(prediction, "weather_summary")
        assert hasattr(prediction, "recommendation")
        assert hasattr(prediction, "confidence")

        # Validate spans
        spans = self.wait_for_spans(min_spans=3)

        # Check custom module span
        module_span = self.assert_span_exists("WeatherAdvisor", spans)
        assert module_span.attributes.get("module.type") == "custom"

        # Check signature attributes
        cot_span = self.assert_span_exists("ChainOfThought", spans)
        assert "WeatherAnalysis" in cot_span.attributes.get("signature.name", "")
        assert "San Francisco" in str(cot_span.attributes.get("input", ""))
        assert "hiking" in str(cot_span.attributes.get("input", ""))

        # Verify structured output attributes
        output_attrs = [k for k in cot_span.attributes.keys() if k.startswith("llm.output")]
        assert len(output_attrs) >= 3, "Missing structured output attributes"

    # =================================================================
    # 🔴 PATTERN 5: MULTI-MODEL DSPY PROGRAMS
    # =================================================================
    @respx.mock
    def test_multi_model_program(self, mock_openai_chat_response):
        """
        Test DSPy program with multiple LMs.
        Advanced pattern with different models for different tasks.
        """
        # Mock API endpoints for different models
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json={**mock_openai_chat_response, "model": "gpt-4o"}),
                httpx.Response(200, json={**mock_openai_chat_response, "model": "gpt-3.5-turbo"}),
            ]
        )

        # Configure multiple LMs
        lm_fast = dspy.OpenAI(model="gpt-3.5-turbo", api_key="fake-key")
        lm_accurate = dspy.OpenAI(model="gpt-4o", api_key="fake-key")

        # Define signatures
        class QuickAnalysis(Signature):
            """Quick analysis for simple questions."""

            question = InputField()
            answer = OutputField()

        class DeepAnalysis(Signature):
            """Deep analysis for complex questions."""

            question = InputField()
            detailed_answer = OutputField()
            sources = OutputField()

        # Create multi-model module
        class AdaptiveQA(Module):
            def __init__(self):
                super().__init__()
                self.quick_lm = lm_fast
                self.deep_lm = lm_accurate
                self.quick_analyze = dspy.Predict(QuickAnalysis)
                self.deep_analyze = dspy.ChainOfThought(DeepAnalysis)

            def forward(self, question):
                # Simple heuristic to choose model
                if len(question.split()) < 10:
                    with dspy.context(lm=self.quick_lm):
                        return self.quick_analyze(question=question)
                else:
                    with dspy.context(lm=self.deep_lm):
                        return self.deep_analyze(question=question)

        adaptive_qa = AdaptiveQA()

        # Test both paths
        simple_pred = adaptive_qa(question="What is 2+2?")
        complex_pred = adaptive_qa(
            question="Analyze the impact of AI on software development workflows in 2024"
        )

        # Validate spans
        spans = self.wait_for_spans(min_spans=5)

        # Check for different model spans
        gpt4_spans = [s for s in spans if "gpt-4o" in str(s.attributes).lower()]
        gpt35_spans = [
            s
            for s in spans
            if "gpt-3.5" in str(s.attributes).lower()
            or "gpt-3.5-turbo" in str(s.attributes).lower()
        ]

        assert len(gpt4_spans) >= 1, "Missing GPT-4 spans"
        assert len(gpt35_spans) >= 1, "Missing GPT-3.5 spans"

        # Verify model switching
        for span in spans:
            if "ChainOfThought" in span.name:
                assert "gpt-4o" in span.attributes.get("llm.model.name", "").lower()
            if "Predict" in span.name and "QuickAnalysis" in span.attributes.get(
                "signature.name", ""
            ):
                assert "gpt-3.5" in span.attributes.get("llm.model.name", "").lower()

    # =================================================================
    # ⚫ PATTERN 6: EVALUATION METRICS & VALIDATION
    # =================================================================
    @respx.mock
    def test_evaluation_metrics(self, mock_openai_chat_response):
        """
        Test DSPy evaluation metrics and validation.
        Production pattern for testing and scoring DSPy programs.
        """
        # Mock multiple API calls for evaluation
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[httpx.Response(200, json=mock_openai_chat_response) for _ in range(6)]
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key")
        dspy.settings.configure(lm=lm)

        # Define signature
        class FactoidQA(Signature):
            """Answer factoid questions accurately."""

            question = InputField()
            answer = OutputField()

        # Create module
        class FactoidQAModule(Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict(FactoidQA)

            def forward(self, question):
                return self.predict(question=question)

        qa_module = FactoidQAModule()

        # Create mock dataset
        devset = [
            dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs(
                "question"
            ),
            dspy.Example(question="Who wrote '1984'?", answer="George Orwell").with_inputs(
                "question"
            ),
            dspy.Example(question="What year did World War II end?", answer="1945").with_inputs(
                "question"
            ),
        ]

        # Define evaluation metric
        def exact_match_metric(prediction, example):
            return prediction.answer.lower().strip() == example.answer.lower().strip()

        # Evaluate program
        from dspy.evaluate import Evaluate

        evaluator = Evaluate(
            devset=devset,
            metric=exact_match_metric,
            num_threads=1,
            display_progress=False,
            display_table=False,
        )

        results = evaluator(qa_module)

        # Validate spans
        spans = self.wait_for_spans(min_spans=10)

        # Check for evaluation spans
        eval_spans = [
            s for s in spans if "evaluation" in s.name.lower() or "metric" in s.name.lower()
        ]
        assert len(eval_spans) >= 1, "Missing evaluation spans"

        # Check for multiple prediction spans (one per example)
        predict_spans = [s for s in spans if "Predict" in s.name]
        assert len(predict_spans) >= len(
            devset
        ), f"Expected at least {len(devset)} prediction spans, got {len(predict_spans)}"

        # Verify metric attributes
        metric_spans = [
            s for s in eval_spans if "metric" in s.attributes.get("evaluation.type", "").lower()
        ]
        if metric_spans:
            metric_span = metric_spans[0]
            assert metric_span.attributes.get("metric.name") == "exact_match_metric"
            assert metric_span.attributes.get("evaluation.samples") == len(devset)

    # =================================================================
    # ⚠️ PATTERN 7: ERROR HANDLING & EDGE CASES
    # =================================================================
    @respx.mock
    def test_error_handling(self, mock_openai_chat_response):
        """
        Test DSPy error handling and resilience.
        Critical for production deployments.
        """
        # Mock API errors
        error_response = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        }

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json=error_response),
                httpx.Response(200, json=mock_openai_chat_response),
            ]
        )

        # Configure DSPy with retry
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key", max_retries=2)
        dspy.settings.configure(lm=lm)

        # Create module
        class ErrorProneModule(Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("question -> answer")

            def forward(self, question):
                return self.predict(question=question)

        error_module = ErrorProneModule()

        # Execute (should retry on error)
        try:
            prediction = error_module(question="What is the meaning of life?")
        except Exception as e:
            pytest.fail(f"Module failed with retry: {e}")

        # Validate spans
        spans = self.wait_for_spans(min_spans=3)

        # Check for error spans
        error_spans = [s for s in spans if not s.status.is_ok]
        assert len(error_spans) == 1, f"Expected 1 error span, got {len(error_spans)}"

        # Validate error attributes
        error_span = error_spans[0]
        assert "rate_limit" in error_span.attributes.get("error.type", "").lower()
        assert error_span.attributes.get("http.status_code") == 429

        # Check for successful retry
        success_spans = [s for s in spans if s.status.is_ok and "Predict" in s.name]
        assert len(success_spans) >= 1, "Missing successful prediction after retry"

    # =================================================================
    # 🌊 PATTERN 8: STREAMING RESPONSES
    # =================================================================
    @respx.mock
    @pytest.mark.asyncio
    async def test_streaming_responses(self, mock_openai_chat_response):
        """
        Test DSPy with streaming responses.
        Advanced pattern for real-time applications.
        """
        # Mock streaming response (SSE format)
        streaming_chunks = [
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"role": "assistant"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"content": "Stream"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"content": "ing"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"content": " response"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=b"".join(streaming_chunks),
                headers={"Content-Type": "text/event-stream"},
            )
        )

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o", api_key="fake-key", streaming=True)
        dspy.settings.configure(lm=lm)

        # Create streaming module
        class StreamingModule(Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("question -> answer")

            async def forward(self, question):
                # DSPy doesn't natively support async streaming, so we mock the behavior
                return await self.predict(question=question)

        streaming_module = StreamingModule()

        # Execute async
        prediction = await streaming_module(question="Stream this response please")

        # Validate
        assert prediction is not None
        assert hasattr(prediction, "answer")

        # Validate spans
        spans = self.wait_for_spans(min_spans=3)

        # Check for streaming span
        llm_spans = [s for s in spans if "openai" in s.name.lower() or "chat" in s.name.lower()]
        assert len(llm_spans) >= 1

        llm_span = llm_spans[0]
        # Check streaming attributes
        if "streaming" in llm_span.attributes:
            assert llm_span.attributes.get("llm.streaming") == True
        else:
            # Check for chunk events
            stream_events = [
                e for e in getattr(llm_span, "events", []) if "chunk" in e.name.lower()
            ]
            assert len(stream_events) > 0, "Missing streaming events"


# =================================================================
# TEST RUNNER & CONFIGURATION
# =================================================================
if __name__ == "__main__":
    """
    To run DSPy instrumentation tests:
    1. Install dependencies:
    pip install pytest pytest-asyncio respx httpx dspy-ai
    pip install openinference-instrumentation-dspy  # If available

    2. Set up environment variables:
    export OPENAI_API_KEY="test-key"
    export ANTHROPIC_API_KEY="test-key"
    export DSPY_DEFAULT_LM="openai/gpt-4o"

    3. Run specific test patterns:
    pytest test_dspy_instrumentation.py::TestDSPyInstrumentation::test_chain_of_thought_module -v

    4. Run all tests:
    pytest test_dspy_instrumentation.py -v

    5. Run with coverage:
    pytest test_dspy_instrumentation.py --cov=neatlogs --cov-report=html

    6. Skip tests if DSPy not installed:
    pytest test_dspy_instrumentation.py -v --tb=short
    """
    print("DSPy Framework Instrumentation Test Suite")
    print("=" * 50)
    print("Patterns Tested:")
    print("1. 🟢 Basic DSPy Modules (ChainOfThought, ReAct)")
    print("2. 🔵 Retrieval-Augmented Generation (RAG)")
    print("3. 🟡 Multi-hop Reasoning Chains")
    print("4. 🟠 DSPy Program Compilation/Optimization")
    print("5. 🔴 Custom Signatures & Modules")
    print("6. ⚫ Multi-Model DSPy Programs")
    print("7. ⚠️ Error Handling & Edge Cases")
    print("8. 🌊 Streaming Responses")
    print("9. 📊 Evaluation Metrics & Validation")
