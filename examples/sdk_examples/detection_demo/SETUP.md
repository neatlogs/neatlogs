# Detection Demo — Setup

Multi-framework workflows that exercise NeatLogs detections (nsfw, hate, jailbreaking, refusals) across LangGraph, CrewAI, LangChain, and Gemini.

## Prerequisites

```bash
cd examples/sdk_examples/detection_demo
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
```

Required env vars:

| Variable | When |
|----------|------|
| `NEATLOGS_API_KEY` | Always |
| `NEATLOGS_ENDPOINT` | Optional (defaults to staging) |
| `AZURE_OPENAI_*` | Workflows 1–4 when `USE_AZURE=true` (default) |
| `OPENAI_API_KEY` | Workflow 4 adversarial classifier scenarios |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Workflow 5 (Gemini streaming) |

## Run

```bash
python main.py                # All workflows (CrewAI skipped if not installed)
python main.py --workflow 1   # Customer Support (LangGraph)
python main.py --workflow 2   # Content Moderation (CrewAI)
python main.py --workflow 3   # Research Assistant (LangChain)
python main.py --workflow 4   # Sales Lead Qualification (LangGraph)
python main.py --workflow 5   # Gemini async streaming
```

## What's simplified

- **No Qdrant / Cohere** — simulated in-memory retrieval and reranking
- **No Docker** — runs against your configured LLM providers
- **PyPI install** — `neatlogs[...]>=1.3.1` from requirements.txt (no editable SDK install)

Traces export to NeatLogs via `NEATLOGS_ENDPOINT`. Open the dashboard to inspect detections and span nesting (`WORKFLOW → AGENT → LLM → RETRIEVER`, etc.).
