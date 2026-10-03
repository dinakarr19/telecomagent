import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from pathlib import Path
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    after_model,
    wrap_tool_call,
    
)

from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

env_path=Path(__file__).resolve().parents[2]/".env"

load_dotenv(env_path)


@dataclass
class BillingContext:
    user_id: str
    account_id: str
    tenant_id:str

class BillingState(AgentState):
    case_id:NotRequired[str]
    model_call_count: NotRequired[int]
    requires_review: NotRequired[bool]


class BillQuery(BaseModel):
    billing_month:str =Field(pattern=r"^\d{4}-\d{2}$",description="Billing month in YYYY-MM format" )
    detail_level:Literal["summary","detailed"]=Field(default="summary",description="Level of detail requested for the bill")
    
    
@tool(args_schema=BillQuery)
def get_bill_summary(
    billing_month:str,
    detail_level:str,
    runtime:ToolRuntime[BillingContext]=None
    ) -> dict:
    """Retrieve the authenticated customer's bill for one month.

    Use for account-specific bill totals, plan charges,
    usage charges, taxes and fees.
    """
    if billing_month not in ["2026-07", "2026-08"]:
        raise ValueError(f"Unsuported billing month: {billing_month}")
    bills = {
        "2026-07": {
            "total": 110.00,
            "plan": 70.00,
            "usage": 25.00,
            "fees_and_taxes": 15.00
        },
        "2026-08": {
            "total": 142.50,
            "plan": 90.00,
            "usage": 35.00,
            "fees_and_taxes": 17.50
        }
    }
    return {
        "tenant_id": runtime.context.tenant_id,
        "account_id": runtime.context.account_id,
        "billing_month": billing_month,
        "detail_level": detail_level,
        "bill": bills[billing_month]
    }

@tool
def calculate_bill_difference(
    earlier_amount:float,
    later_amount:float
    ) -> dict:
    """Calculate the difference between two bill amounts."""
    if earlier_amount < 0 or later_amount < 0:
        raise ValueError("Bill amounts must be non-negative.")
    difference = later_amount - earlier_amount
    
    percentage = (
        difference / earlier_amount * 100
        if earlier_amount
        else None
    )
    return {
        "differernce": round(difference, 2),
        "percentage_change": round(percentage, 2) if percentage is not None else None
    }
    
    
@after_model(state_schema=BillingState)
def increment_model_call_count(
    state: BillingState, 
    runtime)->dict[str, Any]:
    return{
        "model_call_count": state.get("model_call_count", 0) + 1
    }

@wrap_tool_call

def monitor_tool(
    request: ToolCallRequest,
    hanler: Callable[[ToolCallRequest], ToolMessage|Command])-> ToolMessage|Command:
    tool_name=request.tool_call["name"]
    call_id=request.tool_call["id"]
    print(f"tool_start_name={tool_name}, call_id={call_id}")
    
    try:
        result=hanler(request)
        print(f"tool_success_name={tool_name}, call_id={call_id}")
        return result
    except Exception:
        print(f"tool_failure_name={tool_name}, call_id={call_id}")
        raise
    

def format_tool_error(exception:Exception,
                      request:ToolCallRequest)-> str|None:
    if isinstance(exception, ValueError):
        return ( f"Tool {request.tool_call['name']} rejected the input.Correct the arguments  or explain the limitation")
    return None

model_id=os.getenv("MODEL_ID")
if not model_id:
    raise RuntimeError("MODEL_ID environment variable is not set.")

model=init_chat_model(model_id,temperature=0,timeout=30,max_retries=2)
checkpoint_saver=InMemorySaver()

agent=create_agent(
    model=model,
    tools=[get_bill_summary, calculate_bill_difference],
    system_prompt="""
    You are a telecom billing explanation assistant.

    Use tools for all account-specific facts and arithmetic.
    Never invent billing values.
    Clearly separate verified facts from possible causes.
    If information is unavailable, state the limitation.
    Do not perform account changes or financial adjustments.
    """,
    state_schema=BillingState,
    context_schema=BillingContext,
    checkpointer=checkpoint_saver,
    middleware=[
        ToolErrorMiddleware(on_error=format_tool_error),
        ToolRetryMiddleware(
            tools=["get_bill_summary"],
            retry_on=(ConnectionError,TimeoutError),
            max_retries=2,
            initial_delay=1.0,
            backoff_factor=2.0,
            max_delay=5.0,
            jitter=True,
            on_failure="error"),
        ModelCallLimitMiddleware(
            run_limit=4,
            thread_limit=8,
            exit_behavior="error"),
        ToolCallLimitMiddleware(
            run_limit=8,
            thread_limit=20,
            exit_behavior="error"),
        monitor_tool,
        increment_model_call_count])
            
config = {
    "configurable": {
        "thread_id": "billing-session-10482"
    },

    "recursion_limit": 30,

    "tags": [
        "telecom",
        "billing-explanation",
        "day-05"
    ],

    "metadata": {
        "environment": "development"
    }
}
result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": (
                "Compare my July and August bills "
                "and explain the increase."
            )
        }],

        "case_id": "CASE-10482",
        "model_call_count": 0,
        "requires_review": False
    },

    config=config,

    context=BillingContext(
        user_id="authenticated-user-782",
        account_id="A123",
        tenant_id="telecom-us"
    )
)      


final_message = result["messages"][-1]

print(final_message.text)
print("Model calls:", result.get("model_call_count"))
print("Requires review:", result.get("requires_review"))
