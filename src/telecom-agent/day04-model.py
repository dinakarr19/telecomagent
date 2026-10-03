from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from pathlib import Path
from dotenv import load_dotenv
env_path=Path(__file__).resolve().parents[2]/".env"

load_dotenv(env_path)


@dataclass
class BillingContext:
    user_id: str
    account_id: str
    
    
class BillInput(BaseModel):
    billing_month:str =Field(pattern=r"^\d{4}-\d{2}$",description="Billing month in YYYY-MM format" )
    detail_level:Literal["summary","detailed"]=Field(description="Level of detail requested for the bill")


class DifferenceInput(BaseModel):
   earlier_amount:float=Field(ge=0,description="Total bill amount for the earlier month")
   later_amount:float=Field(ge=0,description="Total bill amount for the later month")
   
@tool(args_schema=BillInput)
def get_my_bill(
    billing_month:str,
    detail_level:str="summary",
    runtime:ToolRuntime[BillingContext]=None
    ) -> dict:
     
    """Retrieve the authenticated customer's bill for one month.

    Use for questions about totals, taxes, usage or fees.
    This tool only reads billing information.
    """
    
    account_id=runtime.context.account_id
    user_id=runtime.context.user_id
    
    sample_bills = {
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
    
    bill = sample_bills.get(billing_month)
    if bill is None:
        return {
            "status":"not_found",
            "billing_month":billing_month
        }
    return{
        "status":"found",
        "user_id":user_id,
        "account_id":account_id,
        "billing_month":billing_month,
        "detail_level":detail_level,
        "bill":bill
    }

@tool(args_schema=DifferenceInput)
def calculate_bill_difference(
    earlier_amount:float,
    later_amount:float,
    
    ) -> dict:
    
    """Calculate the difference between two bill amounts and the percentage change."""

  
    
    difference=later_amount-earlier_amount
    percentage_change=(difference/earlier_amount)*100 if earlier_amount>0 else None
    return{
        "difference": round(difference, 2),
        "percentage_change": (
            round(percentage_change, 2)
            if percentage_change is not None
            else None)
    }
        
model=init_chat_model("openai:gpt-5-mini",temperature=0,timeout=30,max_retries=2)

agent=create_agent(
    model=model,
    tools=[get_my_bill,calculate_bill_difference],
    context_schema=BillingContext,
    system_prompt="""
    You are a telecom billing explanation assistant.

    Use get_my_bill for account-specific billing facts.
    Use calculate_bill_difference for arithmetic.
    Never invent billing values.
    Clearly distinguish verified facts from possible explanations.
    If required information is unavailable, state the limitation.
    """
)

result=agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Compare my 2026-07 and 2026-08 bills. "
                "Explain the increase."
            )
        }]
    },
    context=BillingContext(
        user_id="authenticated-user-782",
        account_id="A123"
    )
)

print(result["messages"][-1].text)
