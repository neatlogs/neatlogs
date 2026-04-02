# Marketing Strategy Demo — Neatlogs + CrewAI

A multi-agent marketing strategy workflow that demonstrates **Neatlogs
observability** integrated with **CrewAI** and **Gemini grounded search**.

## What It Does

Three AI agents collaborate to produce a complete marketing strategy:

| Agent | Role | Tools |
|-------|------|-------|
| **Lead Market Analyst** | Researches the company, competitors, and audience | Google Search, Website Analyzer |
| **Chief Marketing Strategist** | Synthesises research into a strategy (can delegate) | Google Search |
| **Creative Content Creator** | Produces campaign ideas and ad copy | — |

The pipeline runs **5 sequential tasks**:

1. **Research** — company, competitors, market trends (with live web search)
2. **Project Understanding** — audience profile and key insights
3. **Marketing Strategy** — structured JSON output (name, tactics, channels, KPIs)
4. **Campaign Idea** — structured JSON output (name, description, audience, channel)
5. **Ad Copy** — structured JSON output (title, body) — depends on tasks 3 & 4

## Neatlogs Features Showcased

- **Auto-instrumented traces** — every agent step, LLM call, and tool invocation
- **Agent thoughts** — Thought / Action / Observation captured automatically
- **Prompt templates** — system and user templates tracked via `PromptTemplate` / `UserPromptTemplate`
- **Gemini tool spans** — nested LLM-inside-TOOL spans from the grounded search
- **Structured output** — Pydantic models captured in trace output
- **Delegation** — strategist can delegate sub-tasks to the analyst
- **Token & cost tracking** — per-agent and crew-wide aggregation
- **Detection rules** — built-in detections fire on errors, high latency, token spikes

### Expected Trace Output (~20-30 spans per run)

```
WORKFLOW: marketing_strategy_workflow
├── CREW: Sequential Crew Execution
│   ├── CREWAI_TASK: research_task
│   │   ├── LLM (Azure GPT-4o) — with Thought/Action/Observation
│   │   ├── TOOL: Search the internet with Google
│   │   │   └── LLM (Gemini 2.0 Flash) — grounded search
│   │   ├── TOOL: Analyze website content
│   │   │   └── LLM (Gemini 2.0 Flash) — grounded search
│   │   └── LLM (Azure GPT-4o) — final answer
│   ├── CREWAI_TASK: project_understanding_task
│   │   └── LLM (Azure GPT-4o)
│   ├── CREWAI_TASK: marketing_strategy_task  → MarketStrategy JSON
│   │   └── LLM (Azure GPT-4o)
│   ├── CREWAI_TASK: campaign_idea_task  → CampaignIdea JSON
│   │   └── LLM (Azure GPT-4o)
│   └── CREWAI_TASK: copy_creation_task  → AdCopy JSON
│       └── LLM (Azure GPT-4o)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual API keys
# Optionally set NEATLOGS_ENDPOINT to target production instead of staging
```

### 3. Run

```bash
python main.py
```

### 4. View traces

Open [app.neatlogs.com](https://app.neatlogs.com) and find the
**marketing-strategy-demo** workflow. Click into any trace to see the
full agent execution timeline, tool calls, thoughts, and structured outputs.

## Customising for a Demo

Edit the `DEMO_INPUTS` dict in `main.py` to use a different company:

```python
DEMO_INPUTS = {
    "customer_domain": "your-customer.com",
    "project_description": "Description of the marketing project...",
}
```

## File Structure

```
marketing_strategy_demo/
├── main.py        # Entry point — inits Neatlogs, runs the crew
├── agents.py      # 3 agents with Neatlogs prompt templates
├── task.py        # 5 tasks with Pydantic structured output
├── crew.py        # Crew assembly and execution
├── tools.py       # Gemini grounded search tools
├── requirements.txt
├── .env.example
└── README.md
```

## Requirements

- Python >= 3.10
- Azure OpenAI access (GPT-4o recommended)
- Google Gemini API key (for grounded search)
- Neatlogs API key (free at [neatlogs.com](https://neatlogs.com))
