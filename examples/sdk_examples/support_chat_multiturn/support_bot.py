"""
SaaS support bot — tools, prompts, and the per-turn function-calling loop.

One instance = one conversation. `history` (the google-genai `contents` list)
persists across turns, so later turns genuinely depend on earlier context (the
model "remembers" the plan the user asked about, the price it quoted, etc.).

Every Gemini call is auto-captured because init() enables the "google_genai"
instrumentation; each tool is a @neatlogs.span(kind="TOOL") child so the trace
shows the agentic work under each turn.
"""

import json

import neatlogs
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"

# The single (already-authenticated) customer in this demo.
USER_ID = "u_dave"

SYSTEM_PROMPT = (
    "You are 'SaaS Genius', a friendly, efficient customer-support AI for a SaaS "
    "product. Help users with subscriptions, pricing, features, and account issues. "
    "You have tools to look up a user's subscription, fetch pricing, check feature "
    "availability, upgrade a plan, and open a support ticket. Keep answers concise and "
    "professional. Use earlier turns of the conversation for context — do not re-ask "
    "what the user already told you.\n"
    f"The current user is already authenticated: their user_id is '{USER_ID}'. "
    "Always pass this user_id to any tool that needs it — never ask the user for it. "
    "When the user asks to upgrade or reports a problem, call the appropriate tool "
    "directly rather than asking for confirmation details you already have."
)


# ---------------------------------------------------------------------------
# Tools — each is a TOOL span; the returned dict is fed back to the model.
# Bodies are simulated (no real backend) but shaped like a real support system.
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", tool_name="get_user_subscription_info",
               description="Look up a user's current subscription and usage")
def get_user_subscription_info(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "current_plan": "Free",
        "renewal_date": None,
        "usage": {"projects": 2, "storage_gb": 0.5},
    }


@neatlogs.span(kind="TOOL", tool_name="get_pricing_plans",
               description="Fetch pricing for a plan and billing cycle")
def get_pricing_plans(plan_name: str, billing_cycle: str = "annual") -> dict:
    catalog = {
        ("Pro", "annual"): 199.99,
        ("Pro", "monthly"): 19.99,
        ("Enterprise", "annual"): 999.99,
    }
    return {
        "plan_name": plan_name,
        "billing_cycle": billing_cycle,
        "price_usd": catalog.get((plan_name, billing_cycle), 0.0),
        "features_highlight": ["Advanced Analytics", "Unlimited Projects", "Priority Support"],
    }


@neatlogs.span(kind="TOOL", tool_name="get_feature_details",
               description="Describe a product feature and which plans include it")
def get_feature_details(feature_name: str) -> dict:
    return {
        "feature_name": feature_name,
        "description": "Tailor your project overview with customizable widgets and layouts.",
        "available_in_plans": ["Pro", "Enterprise"],
    }


@neatlogs.span(kind="TOOL", tool_name="upgrade_subscription",
               description="Upgrade a user's subscription plan")
def upgrade_subscription(user_id: str, new_plan: str, billing_cycle: str = "annual") -> dict:
    return {
        "status": "success",
        "user_id": user_id,
        "old_plan": "Free",
        "new_plan": new_plan,
        "billing_cycle": billing_cycle,
        "confirmation_id": "UPG-67890",
    }


@neatlogs.span(kind="TOOL", tool_name="create_support_ticket",
               description="Open a support ticket for a user's issue")
def create_support_ticket(user_id: str, issue_description: str,
                          severity: str = "medium", related_plan: str = "") -> dict:
    return {
        "status": "success",
        "ticket_id": "TICKET-XYZ-123",
        "severity": severity,
        "estimated_resolution": "24-48 hours",
    }


_TOOL_IMPLS = {
    "get_user_subscription_info": get_user_subscription_info,
    "get_pricing_plans": get_pricing_plans,
    "get_feature_details": get_feature_details,
    "upgrade_subscription": upgrade_subscription,
    "create_support_ticket": create_support_ticket,
}

_TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_user_subscription_info",
        description="Look up a user's current subscription and usage.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"user_id": types.Schema(type="STRING")},
            required=["user_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_pricing_plans",
        description="Fetch pricing for a plan and billing cycle.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "plan_name": types.Schema(type="STRING"),
                "billing_cycle": types.Schema(type="STRING", description="'monthly' or 'annual'"),
            },
            required=["plan_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_feature_details",
        description="Describe a product feature and which plans include it.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"feature_name": types.Schema(type="STRING")},
            required=["feature_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="upgrade_subscription",
        description="Upgrade a user's subscription plan.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "user_id": types.Schema(type="STRING"),
                "new_plan": types.Schema(type="STRING"),
                "billing_cycle": types.Schema(type="STRING"),
            },
            required=["user_id", "new_plan"],
        ),
    ),
    types.FunctionDeclaration(
        name="create_support_ticket",
        description="Open a support ticket for a user's issue.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "user_id": types.Schema(type="STRING"),
                "issue_description": types.Schema(type="STRING"),
                "severity": types.Schema(type="STRING"),
                "related_plan": types.Schema(type="STRING"),
            },
            required=["user_id", "issue_description"],
        ),
    ),
])


class SupportBot:
    """One conversation. `history` persists across turns → real multi-turn context."""

    def __init__(self):
        # google.genai.Client caches transport at construction, so it must be
        # created AFTER neatlogs.init() for auto-instrumentation to attach.
        self.client = genai.Client()
        self.history: list = []  # google-genai `contents`, grows every turn

    def _generate(self, contents):
        return self.client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[_TOOLS],
                temperature=0.3,
            ),
        )

    def ask(self, user_message: str) -> str:
        """Run one user turn: append to history, run the tool-calling loop, return the reply."""
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )

        # Tool-calling loop: keep resolving function calls until the model answers.
        while True:
            response = self._generate(self.history)
            candidate = response.candidates[0]
            self.history.append(candidate.content)

            fn_calls = [p.function_call for p in candidate.content.parts if p.function_call]
            if not fn_calls:
                return response.text or ""

            # Execute each requested tool and feed the results back as one turn.
            tool_parts = []
            for call in fn_calls:
                impl = _TOOL_IMPLS.get(call.name)
                args = dict(call.args or {})
                result = impl(**args) if impl else {"error": f"unknown tool {call.name}"}
                print(f"      ↳ tool {call.name}({json.dumps(args)}) -> {json.dumps(result)}")
                tool_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=call.name, response={"result": result}
                    ))
                )
            self.history.append(types.Content(role="user", parts=tool_parts))
