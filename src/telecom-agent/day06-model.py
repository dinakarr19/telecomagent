import operator
from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import START, END , StateGraph


class BillingState(TypedDict):
    customer_id:str
    current_bill:float
    previous_bill:float
    dispute_reason:str
    
    difference:NotRequired[float]
    severity:NotRequired[str]
    requires_review:NotRequired[bool]
    explanation:NotRequired[str]
    
    audit_steps:Annotated[list[str], operator.add]
    

def calculate_difference(state:BillingState) -> float:
    difference=round(state["current_bill"] - state["previous_bill"], 2 )
    return{
        "difference":difference,
        "audit_steps":["calculated bil difference"]
    }     
    
def assess_risk(state:BillingState) -> dict:
    absolute_difference=abs(state["difference"])
    unclear_reason=len(state["dispute_reason"].strip())<10
    
    requires_review= (absolute_difference>=100 or unclear_reason)
    
    if absolute_difference>=100:
        severity="high"
    elif absolute_difference>=30:
        severity="medium"
    else:
        severity="low"
        
    return{
        "severity":severity,
        "requires_review":requires_review,
        "audit_steps":["assessed risk level"]
    }
    
def choose_resolution(state:BillingState) -> Literal["automatic","review"]:
    return "review" if state["requires_review"] else "automatic" 

def automatic_explanation(state:BillingState) -> dict:
    difference=state["difference"]
    return{
        "explanation":f"The bill changed by ${difference:.2f}. Verify usage, recurring charges, Promotions, and one time fees",
        "audit_steps":["generated automatic explanation"]}
def human_review(state:BillingState) -> dict:
    return{
        "explanation":"Specialist review is required becausee the amount is high or dispute description is insufficient.",
        "audit_steps":["Send dispute for human "]
    }

builder=StateGraph(BillingState)

builder.add_node("calculate_difference", calculate_difference)
builder.add_node("assess_risk", assess_risk)
builder.add_node("automatic_explanation", automatic_explanation)
builder.add_node("human_review", human_review)

builder.add_edge(START, "calculate_difference")
builder.add_edge("calculate_difference", "assess_risk")
builder.add_conditional_edges("assess_risk", choose_resolution,
                 {"automatic":"automatic_explanation",
                  "review":"human_review"})
builder.add_edge("automatic_explanation", END)
builder.add_edge("human_review", END)
graph=builder.compile()

result = graph.invoke(
    {
        "customer_id": "CUST-1001",
        "current_bill": 142.50,
        "previous_bill": 110.00,
        "dispute_reason": (
            "My bill increased even though my usage was similar."
        ),
        "audit_steps": ["Received dispute"],
    }
)

print(result)