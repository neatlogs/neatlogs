# Detection Demo - Quick Setup

## Environment Variables Needed

Copy `.env.example` to `.env` and fill in:

```bash
# REQUIRED (3 keys only!)
NEATLOGS_API_KEY=your_neatlogs_api_key
PROJECT_ID=your_project_id
OPENAI_API_KEY=your_openai_api_key

# OPTIONAL (have defaults)
NEATLOGS_ENDPOINT=https://staging-api.neatlogs.com/api/data/v4/batch
OPENAI_MODEL=gpt-4o
DEBUG=true
```

## Setup Commands

```bash
# 1. Navigate to detection-demo
cd /Users/aadarsh/github/neatlogs-sdk/neatlogs/examples/detection-demo

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install neatlogs SDK from local
pip install -e ../../../

# 5. Configure environment
cp .env.example .env
# Edit .env with your keys

# 6. Run all workflows
python main.py
```

## Run Options

```bash
python main.py                # All 3 workflows (18 scenarios)
python main.py --workflow 1   # Customer Support (LangGraph) only
python main.py --workflow 2   # Content Moderation (CrewAI) only
python main.py --workflow 3   # Research Assistant (LangChain) only
```

## What's Simplified

- **No Qdrant** - Uses simulated in-memory retrieval
- **No Cohere** - Uses simulated reranking
- **No Docker** - Nothing to start locally
- **Staging only** - Traces go directly to staging backend

## What You'll See

- 18 test scenarios across 3 frameworks
- ~9 detection triggers (nsfw, hate, jailbreaking, refusals)
- All span types: WORKFLOW, AGENT, CHAIN, LLM, RETRIEVER, RERANKER, TOOL
- Multi-agent orchestration with proper nesting
