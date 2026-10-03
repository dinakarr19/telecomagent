import os
from dotenv import load_dotenv
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode 
from langgraph.prebuilt.tool_node import tools_condition


from langgraph.graph import START, END, MessagesState,StateGraph

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

model = os.getenv("MODEL_ID")

if not model:
    raise RuntimeError("MODEL_ID not set in .env file") 

@tool
def get_bill_summary(billing_month:str) -> dict:
    
    """Rtreive the customer's bill summary for one month.
    Args:
        billing_month (str): Billing month in YYYY-MM format."""
    bills = {
        "2026-07": {
            "total": 110.00,
            "plan": 70.00,
            "usage": 25.00,
            "fees_and_taxes": 15.00,
        },
        "2026-08": {
            "total": 142.50,
            "plan": 70.00,
            "usage": 55.00,
            "fees_and_taxes": 17.50,
        },
    }
    if billing_month not in bills:
         raise ValueError(f"Unsupported billing month: {billing_month}")
    return{
         "billing_month": billing_month,
         "bill_summary": bills[billing_month]
     }
     
@tool
def calculate_bill_difference(earlier_month: float, later_month: float) -> dict:
    """Calculate the monetary and percentage difference between two billing amounts.
    Args:
        earlier_month (float): The bill amount for the earlier month.
        later_month (float): The bill amount for the later month."""
    if earlier_month <= 0 or later_month <= 0:
        raise ValueError("Bill amounts must be positive numbers.")
    difference = round(later_month - earlier_month, 2)
    percentage_change = (
        round((difference / earlier_month) * 100, 2) if earlier_month != 0 else None
    )
    return {
        "difference": difference,
        "percentage_change": percentage_change,
    }
    
tools=[get_bill_summary,calculate_bill_difference]
model=init_chat_model(model, temperature=0,timeout=30)
model_with_tools=model.bind_tools(tools)

SYSTEM_MESSAGE = SystemMessage(content=("You are a telecom billing assistant. "
        "Use tools for billing facts and calculations. "
        "Never invent customer billing information. "
        "Compare bill components before explaining a change. "
        "Keep the final response concise. "
        "After you have the needed numbers, answer the user directly and do not call more tools."))

def call_model(state:MessagesState)->dict:
    response = model_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    messages = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return END

    ai_tool_call_count = sum(
        1 for msg in messages if getattr(msg, "tool_calls", None)
    )
    return "tools" if ai_tool_call_count <= 1 else END

def format_tool_error(error:Exception)->str:
    return ("The billing tool could not complete the request. "
        "Check the requested month or input values.")
    
tool_node= ToolNode(tools, handle_tool_errors=format_tool_error)

builder = StateGraph(MessagesState)

builder.add_node("assistant", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)
builder.add_edge("tools", "assistant")
agent = builder.compile()

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Compare my July and August 2026 bills. "
                    "Explain why the total increased."
                )
            )
        ]
    },
    config={
        "recursion_limit": 5
    },
)

print(result["messages"][-1].text)
