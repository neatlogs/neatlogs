"""
Marketing Strategy Tasks — five sequential tasks that form the pipeline.

Tasks 3-5 use output_pydantic so results are captured as typed JSON on the
CREWAI_TASK span.

neatlogs.register_crewai_task(task, user_tpl, **vars) attaches a
UserPromptTemplate to each task so the task prompt is visible as a tracked
template on the span when the task executes.
"""

import neatlogs
from neatlogs import UserPromptTemplate
from pydantic import BaseModel, Field
from crewai import Agent, Task


class MarketStrategy(BaseModel):
    name: str = Field(..., description="Name of the marketing strategy")
    tactics: list[str] = Field(..., description="List of specific tactics")
    channels: list[str] = Field(..., description="Marketing channels to use")
    kpis: list[str] = Field(..., description="Key Performance Indicators")


class CampaignIdea(BaseModel):
    name: str = Field(..., description="Campaign name")
    description: str = Field(..., description="What the campaign is about")
    audience: str = Field(..., description="Target audience for this campaign")
    channel: str = Field(..., description="Primary marketing channel")


class AdCopy(BaseModel):
    title: str = Field(..., description="Headline / title of the ad")
    body: str = Field(..., description="Body text of the ad")


def create_tasks(
    analyst: Agent,
    strategist: Agent,
    creator: Agent,
    inputs: dict,
) -> list[Task]:
    customer_domain = inputs.get("customer_domain", "")
    project_description = inputs.get("project_description", "")

    # 1. Research
    research_tpl = UserPromptTemplate(
        "Conduct thorough research about the customer and their competitors "
        "in the context of {{customer_domain}}.\n\n"
        "We are working on this project: {{project_description}}\n\n"
        "Find and analyse:\n"
        "- What the company does, their products/services\n"
        "- Target audience demographics and preferences\n"
        "- Top 3 competitors and their market positioning\n"
        "- Current industry trends and opportunities\n\n"
        "Use the search and website analysis tools to gather real data."
    )
    research_task = Task(
        description=research_tpl.compile(
            customer_domain=customer_domain,
            project_description=project_description,
        ),
        expected_output=(
            "A comprehensive research report covering the company profile, "
            "audience demographics, competitor analysis, and market trends."
        ),
        agent=analyst,
    )
    neatlogs.register_crewai_task(
        research_task, research_tpl,
        customer_domain=customer_domain,
        project_description=project_description,
    )

    # 2. Project understanding
    understanding_tpl = UserPromptTemplate(
        "Review the research findings and develop a deep understanding of "
        "the project and target audience for {{project_description}}.\n\n"
        "Synthesise the research into:\n"
        "- A clear project summary with goals\n"
        "- Detailed target audience profile\n"
        "- Key insights that should shape the marketing strategy"
    )
    project_understanding_task = Task(
        description=understanding_tpl.compile(project_description=project_description),
        expected_output=(
            "A detailed project summary and target audience profile that "
            "will serve as the foundation for the marketing strategy."
        ),
        agent=strategist,
    )
    neatlogs.register_crewai_task(
        project_understanding_task, understanding_tpl,
        project_description=project_description,
    )

    # 3. Marketing strategy (structured output)
    strategy_tpl = UserPromptTemplate(
        "Formulate a comprehensive marketing strategy for "
        "{{customer_domain}} based on all research and audience insights.\n\n"
        "Project: {{project_description}}\n\n"
        "Your strategy must include:\n"
        "- A memorable strategy name\n"
        "- At least 3 specific, actionable tactics\n"
        "- Recommended marketing channels\n"
        "- Measurable KPIs for each tactic"
    )
    marketing_strategy_task = Task(
        description=strategy_tpl.compile(
            customer_domain=customer_domain,
            project_description=project_description,
        ),
        expected_output=(
            "A structured marketing strategy with name, tactics, channels, "
            "and KPIs in the required JSON format."
        ),
        agent=strategist,
        output_pydantic=MarketStrategy,
    )
    neatlogs.register_crewai_task(
        marketing_strategy_task, strategy_tpl,
        customer_domain=customer_domain,
        project_description=project_description,
    )

    # 4. Campaign idea (structured output)
    campaign_tpl = UserPromptTemplate(
        "Develop a creative marketing campaign idea for "
        "{{project_description}}.\n\n"
        "The campaign should:\n"
        "- Be innovative and attention-grabbing\n"
        "- Align with the marketing strategy\n"
        "- Speak directly to the target audience\n"
        "- Be feasible to execute on the recommended channels"
    )
    campaign_idea_task = Task(
        description=campaign_tpl.compile(project_description=project_description),
        expected_output=(
            "A creative campaign idea with name, description, audience, "
            "and channel in the required JSON format."
        ),
        agent=creator,
        output_pydantic=CampaignIdea,
    )
    neatlogs.register_crewai_task(
        campaign_idea_task, campaign_tpl,
        project_description=project_description,
    )

    # 5. Ad copy (structured output, depends on strategy + campaign)
    copy_tpl = UserPromptTemplate(
        "Write compelling marketing copy for the campaign.\n\n"
        "The copy must:\n"
        "- Have a powerful, attention-grabbing headline\n"
        "- Include persuasive body text that drives action\n"
        "- Align with both the marketing strategy and campaign idea\n"
        "- Include a clear call-to-action"
    )
    copy_creation_task = Task(
        description=copy_tpl.compile(),
        expected_output=(
            "Marketing ad copy with a title and body in the required JSON format."
        ),
        agent=creator,
        output_pydantic=AdCopy,
        context=[marketing_strategy_task, campaign_idea_task],
    )
    neatlogs.register_crewai_task(copy_creation_task, copy_tpl)

    return [
        research_task,
        project_understanding_task,
        marketing_strategy_task,
        campaign_idea_task,
        copy_creation_task,
    ]
