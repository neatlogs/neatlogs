"""
Check which HTTP library OpenAI SDK uses
"""
import sys

print("Checking OpenAI SDK HTTP implementation...\n")

try:
    import openai
    print(f"✅ OpenAI SDK version: {openai.__version__}")
    
    # Check if it uses httpx
    try:
        import httpx
        print(f"✅ httpx installed: {httpx.__version__}")
    except ImportError:
        print("❌ httpx not installed")
    
    # Check if it uses requests
    try:
        import requests
        print(f"✅ requests installed: {requests.__version__}")
    except ImportError:
        print("❌ requests not installed")
    
    # Check OpenAI's HTTP client
    from openai import OpenAI
    client = OpenAI(api_key="test")
    print(f"\n📡 OpenAI HTTP Client type: {type(client._client).__name__}")
    print(f"   Module: {type(client._client).__module__}")
    
    # Modern OpenAI SDK (>= 1.0) uses httpx
    if "httpx" in type(client._client).__module__:
        print("\n✅ OpenAI uses HTTPX for HTTP requests")
        print("   → Need to instrument httpx!")
    elif "requests" in type(client._client).__module__:
        print("\n✅ OpenAI uses REQUESTS for HTTP requests")
        print("   → Need to instrument requests!")
    else:
        print(f"\n⚠️  OpenAI uses unknown HTTP library: {type(client._client).__module__}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
