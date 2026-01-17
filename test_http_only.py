"""
Test if HTTP instrumentation works standalone
"""
import neatlogs
import requests

print("Initializing NeatLogs with HTTP tracing...")
neatlogs.init(
    api_key="EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5",
    workflow_name="http-test",
    session_id="http_test_001",
    enable_http_tracing=True,
    instrumentations=[],  # NO framework instrumentations, just HTTP,
    debug=True
)

print("\n🌐 Making HTTP request...")
response = requests.get("https://httpbin.org/get")
print(f"✅ Response: {response.status_code}")

print("\n⏳ Flushing spans...")
neatlogs.flush()

print("\n✅ Check /tmp/kafka-spans-debug.jsonl for HTTP spans!")
print("   Look for otel_kind: 3 or 'CLIENT'")
