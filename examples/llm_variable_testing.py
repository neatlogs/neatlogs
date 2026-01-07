import os
import sys
import time
import logging
import requests
from typing import Dict, List, Any
import anthropic

# --- ENVIRONMENT HOTFIXES ---
try:
    import aiohttp
    if not hasattr(aiohttp, "ConnectionTimeoutError"):
        class ConnectionTimeoutError(Exception): pass
        aiohttp.ConnectionTimeoutError = ConnectionTimeoutError
    if not hasattr(aiohttp, "SocketTimeoutError"):
        class SocketTimeoutError(Exception): pass
        aiohttp.SocketTimeoutError = SocketTimeoutError
except ImportError:
    pass

# Force local neatlogs prioritization
local_path = os.path.join(os.getcwd(), "neatlogs")
if local_path not in sys.path:
    sys.path.insert(0, local_path)

# --- INITIALIZE NEATLOGS ---
import neatlogs
API_KEY = "kL-zN954K3s_lz4FMy9-p0QLqI5S4zFK"
neatlogs.init(api_key=API_KEY, base_url="http://localhost:3000", debug=True)

# 1️⃣ OpenAI (official SDK) — NO native variables

def test_openai_pattern_a():
    print("\n>>> OpenAI Pattern A: Variables in instructions (Responses API)")
    try:
        from openai import OpenAI
        client = OpenAI()
        persona = "You are a strict code reviewer"
        language = "TypeScript"

        # NOTE: client.responses.create is a new/specific API. 
        # If it fails, it demonstrates how variables are resolved before the call.
        client.responses.create(
            model="gpt-4.1-mini",
            instructions=f"{persona}. Review code written in {language}.",
            input="Here is the code: ..."
        )
    except Exception as e:
        print(f"   [RESULT] Pattern A completed. Error (expected if no key/model): {type(e).__name__}")

def test_openai_pattern_b():
    print("\n>>> OpenAI Pattern B: Variables in structured messages")
    try:
        from openai import OpenAI
        client = OpenAI()
        service_name = "payment-gateway"

        client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a performance engineer"},
                {"role": "user", "content": f"Analyze latency for service={service_name}"}
            ]
        )
    except Exception as e:
        print(f"   [RESULT] Pattern B completed. Error: {type(e).__name__}")

# 2️⃣ Anthropic (Claude SDK) — NO native variables

def test_anthropic_pattern_a():
    print("\n>>> Anthropic Pattern A: Variables inside system")
    try:
        client = anthropic.Anthropic()
        tone = "critical"
        domain = "distributed systems"

        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=f"You are a {tone} reviewer for {domain}.",
            messages=[
                {"role": "user", "content": "Review this design doc."}
            ]
        )
    except Exception as e:
        print(f"   [RESULT] Pattern A completed. Error: {type(e).__name__}")

def test_anthropic_pattern_b():
    print("\n>>> Anthropic Pattern B: Variables as structured content blocks")
    try:
        import anthropic
        client = anthropic.Anthropic()
        trace_id = "tr-12345"
        sla_ms = 200

        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this incident."},
                        {"type": "text", "text": f"Trace ID: {trace_id}"},
                        {"type": "text", "text": f"SLA: {sla_ms}ms"}
                    ]
                }
            ]
        )
    except Exception as e:
        print(f"   [RESULT] Pattern B completed. Error: {type(e).__name__}")

def test_anthropic_pattern_c():
    print("\n>>> Anthropic Pattern C: Stable prompt + variable suffix (caching)")
    try:
        import anthropic
        client = anthropic.Anthropic()
        incident_id = "INC-999"

        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an SRE. Follow these rules: ...",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": f"Investigate incident {incident_id}"
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        print(f"   [RESULT] Pattern C completed. Error: {type(e).__name__}")

# 3️⃣ Google GenAI (from google import genai) — NO native variables

def test_google_pattern_a():
    print("\n>>> Google Pattern A: Variables in system instruction")
    # try:
    from google.genai import Client
    client = Client()
    role = "teacher"
    level = "beginner"

    print(client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            f"You are a {role}. Explain vector databases to a {level}."
        ]
    ))
    # except Exception as e:
    #     print(f"   [RESULT] Pattern A completed. Error: {type(e).__name__}")

def test_google_pattern_b():
    print("\n>>> Google Pattern B: Variables as multi-part content")
    try:
        from google.genai import Client
        client = Client()
        user_id = "user_789"
        region = "us-east-1"
        query = "slow database"

        client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Analyze the following request:",
                f"User ID: {user_id}",
                f"Region: {region}",
                f"Query: {query}"
            ]
        )
    except Exception as e:
        print(f"   [RESULT] Pattern B completed. Error: {type(e).__name__}")

# 4️⃣ LiteLLM (direct SDK) — NO native variables

def test_litellm_sdk():
    print("\n>>> LiteLLM SDK: Unified runtime config injection")
    try:
        import litellm
        persona = "security analyst"
        trace_id = "tr-67890"
        vars = {
            "temperature": 0.2,
            "max_tokens": 400
        }

        litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"You are a {persona}"},
                {"role": "user", "content": f"Summarize trace {trace_id}"}
            ],
            **vars
        )
    except Exception as e:
        print(f"   [RESULT] LiteLLM SDK completed. Error: {type(e).__name__}")

# 5️⃣ LiteLLM Proxy Prompt Management — ✅ NATIVE VARIABLES

def test_litellm_proxy_native_vars():
    print("\n>>> LiteLLM Proxy: NATIVE VARIABLES (prompt_variables)")
    try:
        import litellm
        
        # We pass the template in 'messages' and variables in 'extra_body'
        # This is a 'Pure Observability' win because variables stay separate.
        resp = litellm.completion(
            model="openai/gpt-4o",
            api_base="http://localhost:4000",
            messages=[
                {
                    "role": "user", 
                    "content": "Analyze incident {{incident_id}} for {{role}}."
                }
            ]
        )
        print(f"   [RESULT] Proxy Call Successful")
    except Exception as e:
        print(f"   [RESULT] LiteLLM Proxy completed. Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Starting Direct Provider Variable Tests (Consolidated Patterns)...")
    
    # 1. OpenAI
    # test_openai_pattern_a()
    # test_openai_pattern_b()
    
    # 2. Anthropic
    # test_anthropic_pattern_a()
    # test_anthropic_pattern_b()
    # test_anthropic_pattern_c()
    
    # 3. Google
    # test_google_pattern_a()
    # test_google_pattern_b()
    
    # 4. LiteLLM SDK
    # test_litellm_sdk()
    
    # 5. LiteLLM Proxy (Native Vars)
    test_litellm_proxy_native_vars()
    
    print("\nTests completed. Check DEBUG logs for 'Captured raw arguments' and 'Captured messages'.")
    time.sleep(2)