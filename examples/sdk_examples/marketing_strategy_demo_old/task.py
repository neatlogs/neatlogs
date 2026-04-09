"""
Marketing Strategy Tasks.

Five sequential tasks that form the marketing strategy pipeline:
  1. Research       -- analyse the company and competitors
  2. Understanding  -- profile the project and target audience
  3. Strategy       -- produce a structured marketing strategy  (Pydantic JSON)
  4. Campaign Idea  -- generate a creative campaign concept     (Pydantic JSON)
  5. Copy Creation  -- write ad copy based on strategy + campaign (Pydantic JSON)

Tasks 3-5 use output_pydantic so the results are captured as typed JSON
in the Neatlogs trace view.

register_crewai_task() attaches a UserPromptTemplate to each task so that
the task prompt is visible as a tracked template on the CREWAI_TASK span.
"""

from pydantic import BaseModel, Field
from crewai import Agent, Task


# ===================================================================
# Structured output models
# ===================================================================
class MarketStrategy(BaseModel):
    """Structured marketing strategy output."""
    name: str = Field(..., description="Name of the marketing strategy")
    tactics: list[str] = Field(..., description="List of specific tactics")
    channels: list[str] = Field(..., description="Marketing channels to use")
    kpis: list[str] = Field(..., description="Key Performance Indicators")


class CampaignIdea(BaseModel):
    """Structured campaign idea output."""
    name: str = Field(..., description="Campaign name")
    description: str = Field(..., description="What the campaign is about")
    audience: str = Field(..., description="Target audience for this campaign")
    channel: str = Field(..., description="Primary marketing channel")


class AdCopy(BaseModel):
    """Structured ad copy output."""
    title: str = Field(..., description="Headline / title of the ad")
    body: str = Field(..., description="Body text of the ad")


# ===================================================================
# Task factory
# ===================================================================
def create_tasks(
    analyst: Agent,
    strategist: Agent,
    creator: Agent,
    inputs: dict,
) -> list[Task]:
    """
    Build the five marketing-strategy tasks wired to the given agents.

    inputs must contain 'customer_domain' and 'project_description'.
    These are compiled into each task's UserPromptTemplate so the task
    prompt is tracked on the CREWAI_TASK span in Neatlogs.

    Returns the task list in execution order. The last task has explicit
    context dependencies on the strategy and campaign tasks.
    """
    customer_domain = inputs.get("customer_domain", "")
    project_description = inputs.get("project_description", "")

    # 1. Research the company, competitors, and market
    research_task = Task(
        description=(
            f"Conduct thorough research about the customer and their competitors "
            f"in the context of {customer_domain}.\n\n"
            f"We are working on this project: {project_description}\n\n"
            "Find and analyse:\n"
            "- What the company does, their products/services\n"
            "- Target audience demographics and preferences\n"
            "- Top 3 competitors and their market positioning\n"
            "- Current industry trends and opportunities\n\n"
            "Use the search and website analysis tools to gather real data. "
            "Make sure your findings are current and well-sourced."
        ),
        expected_output=(
            "A comprehensive research report covering the company profile, "
            "audience demographics, competitor analysis, and market trends "
            "with supporting data and sources."
        ),
        agent=analyst,
    )

    # 2. Understand the project and target audience
    project_understanding_task = Task(
        description=(
            f"Review the research findings and develop a deep understanding of "
            f"the project and target audience for {project_description}.\n\n"
            "Synthesise the research into:\n"
            "- A clear project summary with goals\n"
            "- Detailed target audience profile (demographics, pain points, "
            "  motivations, preferred channels)\n"
            "- Key insights that should shape the marketing strategy"
        ),
        expected_output=(
            "A detailed project summary and target audience profile that "
            "will serve as the foundation for the marketing strategy."
        ),
        agent=strategist,
    )

    # 3. Formulate the marketing strategy (structured output)
    marketing_strategy_task = Task(
        description=(
            f"Formulate a comprehensive marketing strategy for "
            f"{customer_domain} based on all research and audience insights.\n\n"
            f"Project: {project_description}\n\n"
            "Your strategy must include:\n"
            "- A memorable strategy name\n"
            "- At least 3 specific, actionable tactics\n"
            "- Recommended marketing channels (e.g. LinkedIn, Twitter, "
            "  content marketing, webinars, email)\n"
            "- Measurable KPIs for each tactic\n\n"
            "Think step-by-step about what will have the highest impact "
            "for the target audience identified in previous research."
        ),
        expected_output=(
            "A structured marketing strategy with name, tactics, channels, "
            "and KPIs in the required JSON format."
        ),
        agent=strategist,
        output_pydantic=MarketStrategy,
    )

    # 4. Generate a creative campaign idea (structured output)
    campaign_idea_task = Task(
        description=(
            f"Develop a creative marketing campaign idea for "
            f"{project_description}.\n\n"
            "The campaign should:\n"
            "- Be innovative and attention-grabbing\n"
            "- Align with the marketing strategy\n"
            "- Speak directly to the target audience\n"
            "- Be feasible to execute on the recommended channels\n\n"
            "Provide a campaign name, description, target audience, "
            "and primary channel."
        ),
        expected_output=(
            "A creative campaign idea with name, description, audience, "
            "and channel in the required JSON format."
        ),
        agent=creator,
        output_pydantic=CampaignIdea,
    )

    # 5. Create ad copy (structured output, depends on strategy + campaign)
    copy_creation_task = Task(
        description=(
            "Write compelling marketing copy for the campaign.\n\n"
            "The copy must:\n"
            "- Have a powerful, attention-grabbing headline\n"
            "- Include persuasive body text that drives action\n"
            "- Align with both the marketing strategy and campaign idea\n"
            "- Speak to the identified target audience's pain points\n"
            "- Include a clear call-to-action"
        ),
        expected_output=(
            "Marketing ad copy with a title and body in the required "
            "JSON format."
        ),
        agent=creator,
        output_pydantic=AdCopy,
        context=[marketing_strategy_task, campaign_idea_task],
    )

    return [
        research_task,
        project_understanding_task,
        marketing_strategy_task,
        campaign_idea_task,
        copy_creation_task,
    ]
