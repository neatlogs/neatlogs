# GobbleCube AI Agent — LangGraph Demo

A fully-functioning **LangGraph multi-agent system** that mirrors the architecture of [GobbleCube](https://gobblecube.ai) — an AI-powered "agentic operational layer" for consumer brands in e-commerce and quick commerce.

Built as a **Neatlogs SDK demo** to showcase multi-agent tracing, LLM call instrumentation, and session-level observability.

---

## Architecture

```
User Query
    │
    ▼
GobbsGPT (Supervisor)           ← classifies + routes + synthesises
    │
    ├── Gobbs Edge  (Analytics Agent)     → NLQ-to-SQL + root-cause analysis
    ├── Gobbs Boost (Ad Automation Agent) → bid optimisation + budget allocation
    ├── Gobbs Flow  (Inventory Agent)     → stockout alerts + PO planning
    └── Gobbs Discover (Market Intel)     → trend detection + white-space analysis
```

Each sub-agent generates **3–6 LLM calls** → a single end-to-end query produces **10+ Neatlogs spans**.

---

## File Structure

```
examples/sdk_examples/gobblecube/
├── main.py                  # ← Entry point (run this)
├── config.py                # Settings, NeatLogs init, Azure OpenAI client
├── supervisor.py            # GobbsGPT supervisor graph
├── agent_analytics.py       # Gobbs Edge  — NLQ-to-SQL pipeline
├── agent_ads.py             # Gobbs Boost — ad optimisation
├── agent_inventory.py       # Gobbs Flow  — inventory + PO planning
├── agent_market_intel.py    # Gobbs Discover — market intelligence
├── requirements.txt         # Python dependencies (PyPI neatlogs)
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
cd examples/sdk_examples/gobblecube
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required variables:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://my-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | e.g. `2024-08-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | e.g. `gpt-4o` |
| `NEATLOGS_API_KEY` | Your Neatlogs API key |

### 3. Run all demo scenarios

```bash
python main.py
```

### 4. Run a specific scenario

```bash
python main.py --scenario 1   # Revenue Diagnostic
python main.py --scenario 2   # Ad Campaign Optimisation
python main.py --scenario 3   # Stockout Emergency
python main.py --scenario 4   # Market Opportunity Discovery
python main.py --scenario 9   # Multi-Agent Partial Failure (Ads + Inventory)
```

### 5. Run a custom query

```bash
python main.py --query "What is our share of search on Zepto for protein bars?"
```

---

## Demo Scenarios

| # | Title | Query | Agent |
|---|---|---|---|
| 1 | **Revenue Diagnostic** | "Why did our revenue drop 15% in Mumbai last week?" | Gobbs Edge |
| 2 | **Ad Campaign Optimisation** | "Our ROAS on Blinkit dropped below 2. What should we change?" | Gobbs Boost |
| 3 | **Stockout Emergency** | "Which SKUs are at risk of stocking out in 48 hours?" | Gobbs Flow |
| 4 | **Market Opportunity** | "What are the fastest growing subcategories we should enter?" | Gobbs Discover |
| 9 | **Multi-Agent Partial Failure** | "Our ROAS is tanking... check ad performance and stock levels together." | Gobbs Boost + Gobbs Flow |

---

## Neatlogs Integration

The SDK is configured in `config.py` (called from `main.py` before any LangChain imports):

```python
neatlogs.init(
    api_key=settings.neatlogs_api_key,
    endpoint=settings.neatlogs_endpoint,
    workflow_name="gobblecube",
    tags=["sdk-examples", "gobblecube", "langgraph", "multi-agent"],
    instrumentations=["langchain", "openai", "azure_ai_inference"],
)
```

Each query runs inside a `neatlogs.trace()` workflow session:

```python
with neatlogs.trace(name="gobbs_gpt_query", kind="WORKFLOW", session_id="demo-scenario-1"):
    result = supervisor.invoke(initial_state)
```

Custom spans are applied with `@neatlogs.span()`:

```python
@neatlogs.span(kind="AGENT", name="root_cause_analysis", role="Diagnostic Agent")
def analyze_root_cause(state): ...

@neatlogs.span(kind="TOOL", name="inventory_snapshot", tool_name="gobbs_flow_api")
def check_inventory(state): ...
```

**What you'll see in Neatlogs per query:**

- Top-level workflow span: `gobbs_gpt_query`
  - `gobbs_gpt_classifier` (LLM call)
  - Sub-agent workflow (e.g. `run_analytics_agent`)
    - `intent_recognition` (LLM)
    - `nlq_to_sql` (LLM)
    - `execute_query` (Tool)
    - `root_cause_analysis` (LLM — conditional)
  - `gobbs_gpt_synthesiser` (LLM call)

---

## API Calls

| Node | Type | Real / Dummy |
|---|---|---|
| Intent recognition | Azure OpenAI | **Real** |
| SQL generation | Azure OpenAI | **Real** |
| Query execution | Simulated Antman engine | **Dummy** |
| Root cause analysis | Azure OpenAI | **Real** |
| Bid recommendations | Azure OpenAI | **Real** |
| Budget allocation | Azure OpenAI | **Real** |
| Execution plan | Simulated campaign API | **Dummy** |
| Inventory snapshot | Simulated Gobbs Flow API | **Dummy** |
| Demand analysis | Azure OpenAI | **Real** |
| PO recommendations | Azure OpenAI | **Real** |
| Market data | Simulated market data API | **Dummy** |
| Trend detection | Azure OpenAI | **Real** |
| White-space analysis | Azure OpenAI | **Real** |
| Strategy synthesis | Azure OpenAI | **Real** |
| GobbsGPT classify | Azure OpenAI | **Real** |
| GobbsGPT synthesise | Azure OpenAI | **Real** |
