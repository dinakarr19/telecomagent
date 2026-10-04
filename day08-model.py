import os
from typing import Literal

from dotenv import load_dotenv
from pathlib import Path
from typing_extensions import NotRequired

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

load_dotenv(Path(__file__).resolve().parent / ".env")


class BillingState(MessagesState):
    customer_id: NotRequired[str]
    approval_status: NotRequired[str]
    reviewer: NotRequired[str]


@tool
def get_bill_summary(customer_id: str, billing_month: str) -> dict:
    """Retrieve the authenticated customer's bill for one month.

    Use for account-specific bill totals, plan charges,
    usage charges, taxes and fees.
    """
    sample_bills = {
        "C-1007": {
            "2026-09": {
                "plan": 65.00,
                "usage": 52.00,
                "fees": 4.20,
                "total": 121.20,
            }
        }
    }

    bill = sample_bills.get(customer_id, {}).get(billing_month)
    if bill is None:
        return {
            "status": "not found",
            "customer_id": customer_id,
            "month": billing_month,
        }
    return {
        "status": "ok",
        "customer_id": customer_id,
        "month": billing_month,
        "bill": bill,
    }


@tool
def apply_bill_credit(customer_id: str, amount: float, reason: str) -> dict:
    """Apply a credit to the customer's account."""
    if amount <= 0:
        return {
            "status": "rejected",
            "reason": "Credit amount must be positive.",
        }
    return {
        "status": "applied",
        "customer_id": customer_id,
        "credited_amount": round(amount, 2),
        "reason": reason,
    }


tools = [get_bill_summary, apply_bill_credit]
tool_node = ToolNode(tools)
SENSITIVE_TOOLS = ["apply_bill_credit"]

model_id = os.getenv("MODEL_ID")
if not model_id:
    raise RuntimeError("MODEL_ID not set in .env file")
model = init_chat_model(model_id, temperature=0, timeout=30)

model_with_tools = model.bind_tools(tools)

SYSTEM_MESSAGE = SystemMessage(
    content=(
        """
You are a telecom billing assistant.

Use tools for customer-specific billing facts.
Never invent billing information.

The apply_bill_credit tool changes customer billing data.
You may propose that tool, but the workflow requires human
approval before it is executed.

Request only one tool at a time.
"""
    )
)


def call_model(state: BillingState):
    response = model_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]])
    return {"messages": [response]}


def route_after_agent(
    state: BillingState,
) -> Literal["end", "review", "tools"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        return "end"
    has_sensitive_tool = any(
        call["name"] in SENSITIVE_TOOLS for call in tool_calls
    )
    if has_sensitive_tool:
        return "review"
    return "tools"


def review_sensitive_action(state: BillingState) -> Command:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    approval_request = {
        "question": "Approve the proposed billing action?",
        "tool_calls": tool_calls,
    }

    decision = interrupt(approval_request)
    action = decision.get("action")
    reviewer = decision.get("reviewer", "unknown")

    if action == "approve":
        return Command(
            update={"approval_status": "approved", "reviewer": reviewer},
            goto="tools",
        )

    rejection_messages = [
        ToolMessage(
            content=(
                "The proposed billing action was rejected by the reviewer."
            ),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        for tool_call in tool_calls
    ]

    return Command(
        update={"approval_status": "rejected", "reviewer": reviewer},
        goto="agent",
        messages=rejection_messages,
    )


builder = StateGraph(BillingState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("review", review_sensitive_action)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "review": "review",
        "tools": "tools",
        "end": END,
    },
)
builder.add_edge("tools", "agent")
builder.add_edge("review", "agent")

checkpoint_saver = InMemorySaver()

agent = builder.compile(checkpointer=checkpoint_saver)

config = {
    "configurable": {
        "thread_id": "billing-case-C-1007-2026-091"
    }
}
result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Show the September 2026 bill "
                    "for customer C-1007."
                )
            )
        ],
        "customer_id": "C-1007",
    },
    config=config,
)

print(result["messages"][-1].content)
paused_result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Apply a $20 courtesy credit because "
                    "the customer experienced a service outage."
                )
            )
        ]
    },
    config=config,
)
if "__interrupt__" in paused_result:
    pending = paused_result["__interrupt__"]

    for item in pending:
        print(item.value)
print("hello")