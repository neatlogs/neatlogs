"""
Example 8: DSPy Programs

DSPy is a framework for algorithmically optimizing LM prompts and weights.
Both OpenInference and OpenLLMetry support DSPy instrumentation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown

# DSPy imports
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric


def example_basic_program():
    """Basic DSPy program with Chain of Thought."""
    print("="*60)
    print("Basic DSPy Program:")
    print("="*60)
    
    # Define a simple QA program
    class QuestionAnswer(dspy.Signature):
        """Answer questions with short factual answers."""
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")
    
    # Create program with Chain of Thought
    qa_program = dspy.ChainOfThought(QuestionAnswer)
    
    # Run it
    with trace("dspy_qa", kind="CHAIN") as span:
        question = "What is the capital of France?"
        prediction = qa_program(question=question)
        
        span.set_attribute("question", question)
        span.set_attribute("answer", prediction.answer)
        
        print(f"Question: {question}")
        print(f"Answer: {prediction.answer}")
        print(f"Reasoning: {prediction.rationale}")


def example_multi_hop_reasoning():
    """Multi-hop reasoning with DSPy."""
    print("\n" + "="*60)
    print("Multi-Hop Reasoning:")
    print("="*60)
    
    class GenerateSearchQuery(dspy.Signature):
        """Write a simple search query for answering a complex question."""
        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        query = dspy.OutputField()
    
    class GenerateAnswer(dspy.Signature):
        """Answer questions with short factual answers."""
        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")
    
    class MultiHopQA(dspy.Module):
        def __init__(self, passages_per_hop=3):
            super().__init__()
            self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
            self.passages_per_hop = passages_per_hop
        
        def forward(self, question):
            context = []
            
            # First hop
            with trace("hop_1", kind="RETRIEVER"):
                query = self.generate_query(
                    context=context,
                    question=question
                ).query
                # In real app, would retrieve passages here
                context.append(f"Search result for: {query}")
            
            # Second hop
            with trace("hop_2", kind="RETRIEVER"):
                query = self.generate_query(
                    context=context,
                    question=question
                ).query
                context.append(f"Search result for: {query}")
            
            # Generate answer
            with trace("generate_final_answer", kind="LLM"):
                answer = self.generate_answer(
                    context=context,
                    question=question
                )
            
            return answer
    
    # Run multi-hop QA
    with trace("multi_hop_qa", kind="CHAIN"):
        program = MultiHopQA()
        question = "Who won the FIFA World Cup in the year the Eiffel Tower was completed?"
        result = program(question=question)
        
        print(f"Question: {question}")
        print(f"Answer: {result.answer}")
    
    # Trace hierarchy shows:
    # - CHAIN span (multi_hop_qa)
    #   └─ RETRIEVER span (hop_1)
    #      └─ LLM span (generate query)
    #         └─ HTTP span
    #   └─ RETRIEVER span (hop_2)
    #      └─ LLM span (generate query)
    #         └─ HTTP span
    #   └─ LLM span (generate_final_answer)
    #      └─ HTTP span


def example_with_optimizer():
    """DSPy with prompt optimization."""
    print("\n" + "="*60)
    print("DSPy Optimizer Example:")
    print("="*60)
    
    # Load training data
    gsm8k = GSM8K()
    trainset = gsm8k.train[:5]  # Small sample for demo
    
    # Define program
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")
        
        def forward(self, question):
            return self.prog(question=question)
    
    # Compile with optimizer
    with trace("dspy_compile", kind="CHAIN") as span:
        student = CoT()
        
        # Use BootstrapFewShot optimizer (simplified for demo)
        # In real app, would use teleprompter = dspy.teleprompt.BootstrapFewShot()
        # compiled = teleprompter.compile(student, trainset=trainset)
        
        span.set_attribute("trainset_size", len(trainset))
        print("Program compiled with optimizer")
        print(f"Training set size: {len(trainset)}")
    
    # Each optimization step is traced:
    # - LLM calls for generating examples
    # - Evaluation metrics
    # - Token usage per optimization step


def main():
    # Enable span logging
    os.environ['NEATLOGS_LOG_SPANS'] = 'true'
    
    # Initialize
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="dspy-programs",
        instrumentations=["dspy", "openai"],
        debug=True,
    )
    
    try:
        with trace("dspy_examples"):
            # Configure DSPy with OpenAI
            lm = dspy.OpenAI(model="gpt-4o-mini")
            dspy.settings.configure(lm=lm)
            
            # Run examples
            example_basic_program()
            example_multi_hop_reasoning()
            example_with_optimizer()
        
        print("\n✅ All DSPy examples completed!")
        print("Check spans.log to see:")
        print("  - Chain of Thought reasoning traces")
        print("  - Multi-hop retrieval patterns")
        print("  - Optimization iterations (if run)")
        print("  - Token usage for each DSPy module")
    except Exception as e:
        print(f"\nError during DSPy execution: {e}")
    finally:
        print("\n💾 Flushing spans...")
        flush()
        print("🛑 Shutting down SDK...")
        shutdown()
        print("✅ Done!")


if __name__ == "__main__":
    main()
