"""
Example 12: Metrics Export Test

Demonstrates:
1. Histogram metrics export (TTFT, streaming latency, tokens/sec)
2. Dual storage: span attributes + OTel metrics
3. OpenLLMetry standard metric names

This example tests that metrics are:
- Recorded to histograms during span processing
- Exported to backend every 60 seconds (or on flush/shutdown)
- Also stored as span attributes (neatlogs.metrics.*) for per-request analysis
"""

import os
import time
from openai import OpenAI

# Import Neatlogs SDK
import sys
sys.path.insert(0, "/Users/tanishabanik/Projects/neatlogs/neatlogs/sdk")
from neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown

# Set environment variables
os.environ["NEATLOGS_LOG_SPANS"] = "true"
os.environ["NEATLOGS_LOG_METRICS"] = "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Neatlogs with metrics export
init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    workflow_name="metrics-test",
    instrumentations=["openai"],
    debug=True,
    metrics_export_interval=10.0,  # Export every 10 seconds for faster testing
)

print("\n" + "=" * 80)
print("🧪 Testing Metrics Export")
print("=" * 80)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Test 1: Streaming response (should generate all metrics)
print("\n📊 Test 1: Streaming Response (TTFT, streaming latency, output tokens/sec)")
print("-" * 80)

with trace(
    name="streaming_test",
    kind="WORKFLOW",
    prompt_template="Tell me about {topic}",
    prompt_variables={"topic": "quantum computing"},
):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Tell me about quantum computing in 3 sentences"}
        ],
        stream=True,
        temperature=0.7,
        max_tokens=100,
    )
    
    result = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            result += chunk.choices[0].delta.content
    
    print(f"\n✓ Streaming response received ({len(result)} chars)")

# Wait a bit to ensure metrics are collected
time.sleep(2)

# Test 2: Non-streaming response (should generate total duration and tokens/sec)
print("\n📊 Test 2: Non-Streaming Response (duration, total tokens/sec)")
print("-" * 80)

with trace(
    name="non_streaming_test",
    kind="WORKFLOW",
    prompt_template="What is {concept}?",
    prompt_variables={"concept": "artificial intelligence"},
):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is artificial intelligence? Answer in 2 sentences"}
        ],
        stream=False,
        temperature=0.7,
        max_tokens=100,
    )
    
    print(f"\n✓ Non-streaming response received: {response.choices[0].message.content[:100]}...")

# Wait a bit to ensure metrics are collected
time.sleep(2)

# Test 3: Multiple requests for histogram distribution
print("\n📊 Test 3: Multiple Requests (for histogram distribution)")
print("-" * 80)

for i in range(3):
    with trace(
        name=f"batch_test_{i+1}",
        kind="WORKFLOW",
    ):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Count to {i+3}"}
            ],
            stream=True,
            max_tokens=50,
        )
        
        for chunk in response:
            pass  # Consume stream
        
        print(f"  ✓ Request {i+1}/3 completed")

print("\n" + "=" * 80)
print("📤 Flushing and shutting down...")
print("=" * 80)

# Flush to send all pending data
flush()

# Wait for metrics export interval (10 seconds) to trigger
print("\n⏳ Waiting 12 seconds for metrics to be exported...")
time.sleep(12)

# Shutdown to finalize
shutdown()

print("\n" + "=" * 80)
print("✅ Metrics Test Complete!")
print("=" * 80)
print("""
Check the following:

1. **Span Attributes** (in spans.log):
   - neatlogs.metrics.time_to_first_token
   - neatlogs.metrics.streaming_latency
   - neatlogs.metrics.output_tokens_per_second
   - neatlogs.metrics.tokens_per_second
   - llm.is_streaming

2. **OTel Metrics** (in metrics.log):
   - gen_ai.server.time_to_first_token (histogram, seconds)
   - llm.chat_completions.streaming_time_to_generate (histogram, seconds)
   - gen_ai.client.operation.duration (histogram, seconds)
   - neatlogs.llm.output_tokens_per_second (histogram, tokens/s)
   - neatlogs.llm.tokens_per_second (histogram, tokens/s)
   - Each metric has: count, sum, min, max, attributes

3. **Metric Attributes** (for span correlation):
   - trace_id: 32-char hex (matches span's trace_id)
   - span_id: 16-char hex (matches span's span_id)
   - gen_ai.response.model: Model name
   - server.address: API endpoint
   
   💡 You can JOIN metrics with spans in backend using trace_id + span_id

4. **Console Output**:
   - Should see "✓ Metrics exported successfully (X data points)" in debug logs
   - Performance stats showing spans processed and exported

5. **Backend**:
   - POST /api/metrics endpoint should receive histogram data
   - Backend can correlate metrics with spans for detailed analysis

📂 Log Files:
   - spans.log (span attributes per request)
   - metrics.log (histogram data points with trace_id/span_id for correlation)

🔍 Example Correlation Query (ClickHouse):
   SELECT m.sum AS ttft_seconds, s.name, s.attributes['llm.request.messages']
   FROM metrics m JOIN spans s 
   ON m.attributes['trace_id'] = s.trace_id 
   AND m.attributes['span_id'] = s.span_id
   WHERE m.metric_name = 'gen_ai.server.time_to_first_token'
   ORDER BY m.sum DESC LIMIT 10;
""")
