# SDK Examples

Runnable reference apps showing NeatLogs instrumentation patterns. Each example installs the SDK from PyPI (`neatlogs[...]>=1.3.1`).

**To add NeatLogs to your own codebase**, install the [neatlogs-py AI skill](https://github.com/neatlogs/skills) and ask your coding agent to instrument your project — don't copy-paste from the root README.

## Quick start (any example)

```bash
cd examples/sdk_examples/<example-name>
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
python main.py         # or python react_agent.py for langchain_react
```

Get `NEATLOGS_API_KEY` from the [NeatLogs dashboard](https://app.neatlogs.com).

## Examples

| Folder | Framework | Entry command |
|--------|-----------|---------------|
| [`anthropic_multiagent/`](anthropic_multiagent/) | Anthropic via Bedrock | `python main.py` |
| [`openai_multiagent/`](openai_multiagent/) | OpenAI via Azure | `python main.py` |
| [`google_genai_multiagent/`](google_genai_multiagent/) | Google GenAI | `python main.py` |
| [`langchain_react/`](langchain_react/) | LangChain ReAct + Bedrock | `python react_agent.py` |
| [`langgraph_multiagent/`](langgraph_multiagent/) | LangGraph multi-provider | `python main.py` |
| [`langgraph_research_assistant/`](langgraph_research_assistant/) | LangGraph routing assistant | `python main.py` |
| [`marketing_strategy_demo/`](marketing_strategy_demo/) | CrewAI + Gemini search | `python main.py` |
| [`neatlogs_support_bot/`](neatlogs_support_bot/) | CrewAI RAG support bot | `python main.py` |
| [`reasoning_model_workflow/`](reasoning_model_workflow/) | Multi-provider reasoning params | `python main.py` |
| [`detection_demo/`](detection_demo/) | Multi-framework detection scenarios | `python main.py` |
| [`support_copilot_demo/`](support_copilot_demo/) | Support agent demo (3 trace stories) | `RUN=A python support_copilot.py` |
| [`support_copilot_demo_triaged/`](support_copilot_demo_triaged/) | Same demo after Triage fixes | `SENDGRID_FAKE_SUCCESS=1 RUN=B python support_copilot.py` |

See [`marketing_strategy_demo/README.md`](marketing_strategy_demo/README.md) for a detailed walkthrough of the CrewAI + Gemini trace shape.

See [`detection_demo/SETUP.md`](detection_demo/SETUP.md) for workflow-by-workflow setup and run options.
