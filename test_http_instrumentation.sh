#!/bin/bash
set -e

echo "======================================================================"
echo "🔍 Testing HTTP Instrumentation"
echo "======================================================================"

cd /Users/tanishabanik/Projects/neatlogs

echo ""
echo "Step 1: Check which HTTP library OpenAI uses..."
echo "----------------------------------------------------------------------"
uv run python check_openai_http.py

echo ""
echo ""
echo "Step 2: Run test with HTTP tracing enabled..."
echo "----------------------------------------------------------------------"
echo "(Look for HTTP instrumentation logs)"
uv run python test_ai_agents_complete.py 2>&1 | grep -E "(HTTP|http|Instrumented)" || true

echo ""
echo ""
echo "Step 3: Check debug log for HTTP spans..."
echo "----------------------------------------------------------------------"
if [ -f /tmp/kafka-spans-debug.jsonl ]; then
    echo "Checking for HTTP spans in debug log:"
    cat /tmp/kafka-spans-debug.jsonl | tail -3 | jq -r '.spans[] | select(.http_method != null or .otel_kind != null) | "Found: \(.name) | otel_kind=\(.otel_kind) | http=\(.http_method) \(.http_url)"' || echo "❌ No HTTP spans found"
    
    echo ""
    echo "All otel_kind values:"
    cat /tmp/kafka-spans-debug.jsonl | jq -r '.spans[].otel_kind' | sort | uniq -c
else
    echo "❌ Debug log not found at /tmp/kafka-spans-debug.jsonl"
fi

echo ""
echo "======================================================================"
echo "✅ Test complete. Summary:"
echo "----------------------------------------------------------------------"
echo "1. If OpenAI uses HTTPX → HTTP instrumentation should work"
echo "2. If you see 'Instrumented httpx' → instrumention is active"
echo "3. If otel_kind shows numbers (1=SERVER, 3=CLIENT) → HTTP spans arriving"
echo "4. If all are null → HTTP spans NOT being sent by SDK"
echo "======================================================================"
