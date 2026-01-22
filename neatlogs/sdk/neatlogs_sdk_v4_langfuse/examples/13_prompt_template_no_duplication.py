"""
Example 13: Universal PromptTemplate - NO Variable Duplication!

This example demonstrates the NEW PromptTemplate approach that eliminates
the duplication problem where you had to specify variables twice.

BEFORE (Old Way - Duplication):
    with trace("query",
               prompt_template="Weather in {city} on {date}",
               prompt_variables={"city": "SF", "date": "Jan 21"}):  # ❌ First time
        llm.invoke({"city": "SF", "date": "Jan 21"})               # ❌ Second time

AFTER (New Way - NO Duplication):
    template = PromptTemplate("Weather in {{city}} on {{date}}")

    with trace("query", prompt_template=template):
        prompt = template.compile(city="SF", date="Jan 21")  # ✅ ONCE!
        llm.invoke(prompt)

Key Benefits:
1. ✅ Variables specified ONCE in template.compile()
2. ✅ Works universally across ALL frameworks (OpenAI, Anthropic, LangChain, etc.)
3. ✅ Auto-captured via ContextVars for tracing
4. ✅ Supports streaming, async, structured outputs
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown, PromptTemplate
from openai import OpenAI
from anthropic import Anthropic
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["NEATLOGS_LOG_SPANS"] = "true"
os.environ["NEATLOGS_LOG_METRICS"] = "true"


def demo_1_string_template():
    """Demo 1: Simple string template"""
    print("\n" + "="*70)
    print("DEMO 1: String Template - No Duplication (OpenAI)")
    print("="*70)

    template = PromptTemplate("Explain {{concept}} to a {{audience}} in {{words}} words")

    print(f"\n📝 Template: {template}")
    print(f"📊 Variables: {template.variables}")

    with trace("explain_query", prompt_template=template):
        prompt_text = template.compile(
            concept="quantum computing",
            audience="beginner",
            words="50"
        )

        print(f"\n✅ Compiled prompt:\n{prompt_text}")

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=100
        )
        
        print(f"🤖 Response: {response.choices[0].message.content[:100]}...")


def demo_2_chat_messages():
    """Demo 2: Chat message template"""
    print("\n" + "="*70)
    print("DEMO 2: Chat Messages - No Duplication (Anthropic)")
    print("="*70)

    template = PromptTemplate([
        {"role": "system", "content": "You are a {{role}}. {{instructions}}"},
        {"role": "user", "content": "Help me with: {{task}}"}
    ])

    print(f"\n📝 Template: {template}")
    print(f"📊 Variables: {template.variables}")

    with trace("assistant_query", prompt_template=template):
        messages = template.compile(
            role="helpful assistant",
            instructions="Provide concise answers.",
            task="understanding Python decorators"
        )

        print(f"\n✅ Compiled messages:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")

        client = Anthropic()
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        user_msgs = [m for m in messages if m["role"] == "user"]
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            system=system_msg,
            messages=user_msgs,
            max_tokens=100
        )
        
        print(f"🤖 Response: {response.content[0].text[:100]}...")


def demo_3_missing_variables():
    """Demo 3: Error handling for missing variables"""
    print("\n" + "="*70)
    print("DEMO 3: Validation - Missing Variables")
    print("="*70)

    template = PromptTemplate("Query {{query}} using {{context}} and {{tool}}")

    print(f"\n📝 Template requires: {template.variables}")

    with trace("validation_test", prompt_template=template):
        try:
            # Try to compile with missing variables
            template.compile(query="What is AI?")  # ❌ Missing context and tool
        except ValueError as e:
            print(f"\n✅ Caught error: {e}")
            print("   Template validation prevents missing variables!")


def demo_4_universal_compatibility():
    """Demo 4: Universal compatibility demo"""
    print("\n" + "="*70)
    print("DEMO 4: Universal Compatibility (LangChain)")
    print("="*70)

    template = PromptTemplate("Translate '{{text}}' to {{language}}")

    print(f"\n📝 Template: {template}")

    with trace("universal_demo", prompt_template=template):
        prompt_text = template.compile(
            text="Hello world",
            language="Spanish"
        )

        print(f"\n✅ Compiled: {prompt_text}")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        response = llm.invoke(prompt_text)
        
        print(f"🤖 Response: {response.content}")


def demo_5_comparison():
    """Demo 5: Direct comparison - Old vs New"""
    print("\n" + "="*70)
    print("DEMO 5: Side-by-Side Comparison (OpenAI)")
    print("="*70)

    city = "San Francisco"
    date = "January 21, 2025"

    print("\n❌ OLD WAY (Duplication):")
    print("-" * 70)
    print("with trace(")
    print("    'weather_query',")
    print(f"    prompt_template='Weather in {{{{city}}}} on {{{{date}}}}',")
    print(f"    prompt_variables={{'city': '{city}', 'date': '{date}'}}  # ❌ First time")
    print("):")
    print(f"    llm.invoke({{'city': '{city}', 'date': '{date}'}})      # ❌ Second time")

    print("\n✅ NEW WAY (No Duplication):")
    print("-" * 70)
    print("template = PromptTemplate('Weather in {{city}} on {{date}}')")
    print("")
    print("with trace('weather_query', prompt_template=template):")
    print(f"    prompt = template.compile(city='{city}', date='{date}')  # ✅ ONCE!")
    print("    llm.invoke(prompt)")

    template = PromptTemplate("Weather in {{city}} on {{date}}")

    with trace("weather_query_new", prompt_template=template):
        prompt_text = template.compile(city=city, date=date)
        print(f"\n✅ Compiled: {prompt_text}")

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=50
        )
        
        print(f"🤖 Response: {response.choices[0].message.content[:100]}...")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🚀 PROMPT TEMPLATE - NO DUPLICATION EXAMPLES")
    print("="*70)
    print("\nDemonstrating the NEW PromptTemplate approach that eliminates")
    print("variable duplication across all LLM frameworks.")

    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="prompt-template-examples",
        instrumentations=["openai", "anthropic", "langchain"],
        debug=True,
    )

    # Run demos
    demo_1_string_template()
    demo_2_chat_messages()
    demo_3_missing_variables()
    demo_4_universal_compatibility()
    demo_5_comparison()

    print("\n" + "="*70)
    print("🎉 SUMMARY")
    print("="*70)
    print("✅ Variables specified ONCE in template.compile()")
    print("✅ Auto-captured via ContextVars for tracing")
    print("✅ Works universally across ALL frameworks")
    print("✅ Tested with OpenAI, Anthropic, and LangChain")
    print("="*70 + "\n")

    # Cleanup
    flush()
    shutdown()


if __name__ == "__main__":
    main()
