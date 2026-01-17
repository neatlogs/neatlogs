from agnost import track, config

track(server, "15c1465f-7d05-4c20-bacb-290d7f472e7b", config(
    endpoint="https://api.agnost.ai"
))

import os
import raindrop.analytics as raindrop

# Recommended: load from env var
raindrop.init("311070a8-8f27-46fe-b548-870c963a46fa")
raindrop.set_debug_logs(True)

import uuid
import os
from openai import OpenAI

message = "What is love?"
event_id = str(uuid.uuid4())  # correlate logs across systems

interaction = raindrop.begin(event_id=event_id,
                              event='chat_message',
                              user_id='user_123',
                              input=message,
                              convo_id='convo_123')  # omit if not conversational

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": message}]
)
text = response.choices[0].message.content

interaction.finish(output=text)
raindrop.flush()
raindrop.shutdown()

from phoenix.otel import register

tracer_provider = register(
    project_name="support-bot",
    auto_instrument=True
)

from langsmith import traceable
from langfuse import get_client