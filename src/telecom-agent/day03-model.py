from typing import Literal
from pydantic import BaseModel, Field
from pydantic import ValidationError
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage


from pathlib import Path
env_path=Path(__file__).resolve().parents[2]/".env"

load_dotenv(env_path)
model_id=os.getenv("MODEL_ID")

if not model_id:
    raise RuntimeError("Model id is not configured")

model= init_chat_model(
    model_id,
    temperature =0,
    timeout=30,
    max_retries = 2
)

class BillingAssessment(BaseModel):
    """Structured assessment of a telecom billing question"""
    intent:Literal["bill_increase",
    "promotion_question",
    "general_billing_question",
    "other"] =Field(description= "The customer's primary billing intent")
    account_specific_answer_possible: bool = Field(description = "Whether the question can be answereed using the info currently available")
    possible_causes:list[str]=Field(description="Possible causes that may be presented as verified customer facts")
    required_data:list[str]=Field(description="Account data required to verify the answer")
    confidence:float = Field(description = "Confidence in the intent classification",ge=0.0, le=1.0)
    recommended_next_step:Literal["explain_general_concept","require_account_context","request_clarification","route_for_authorization"]=Field(
        description="The safest next step")
    urgency:Literal["low","medium","high"]=Field(description="Urgency of the request")
    requires_human_review:bool=Field(description="Resolution requires Human Review")
    reason_for_human_review:str|None = Field(description="Specify the reason  for human review",default=None)

structured_model= model.with_structured_output(
    BillingAssessment,
    include_raw=True
)


system_message = SystemMessage(
    """
    You are a telecom billing-assessment assistant.

    No real customer account information is available.

    Do not claim that a possible cause is verified.

    Classify the request and identify the information required
    to verify an account-specific answer.
    """
)

def assess_question(question:str)->BillingAssessment:
    result= structured_model.invoke(
        [
            system_message,
            HumanMessage(content=question)
        ],
        config={
            "run_name":"day03-billing-assessment",
            "tags":[
                "day03",
                "structured_output"
            ],
            "metadata":{
                "schema":"BillingAssessment",
                "schema_version":"1.0"
            }
        }
    )
    
    if result["parsing_error"] is not None:
        raise RuntimeError(
            "No structured assessment was returned"
        )
    parsed_result=result["parsed"]
    if parsed_result is None:
        raise RuntimeError(
            "No structured assessment was returned"
        )
    return parsed_result
    
if __name__ == "__main__":
    assessment = assess_question(
        "Apply a $50 credit."
    )
    if (
    assessment.intent == "credit_request"
    or assessment.confidence < 0.70
):
        assessment.reason_for_human_review=True

    print(assessment)
    print()
    print(assessment.model_dump(mode="json"))