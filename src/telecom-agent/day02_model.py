import asyncio
import os

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate



from pathlib import Path
env_path=Path(__file__).resolve().parents[2]/".env"

load_dotenv(env_path)
model_id=os.getenv("MODEL_ID")

if not model_id:
    raise RuntimeError("Model id is not configured")

model = init_chat_model(
    model_id,
    temperature=0,
    timeout = 30,
    max_retries = 2
)

system_message=SystemMessage(
    content = """
    You are a telecom billing assistant.

    You currently have no access to the customer's real account.

    Never invent account-specific billing information.

    When information is unavailable:
    1. state the limitation;
    2. identify the information required;
    3. explain possible causes without presenting them as facts.

    Respond in at most five sentences.
    """
)

def ask(question:str)->str:
    """Invoke the Model Synchronously"""
    
    messages = [ system_message,HumanMessage(content=question)]
    
    response = model.invoke(
        messages,
        config= {
            "run_name":"day02-single-question",
            "tags":["day02","telecom-billing"],
            "metadata":{
                "lesson":"lanhchain-foundations"
            }
        }
    )
    return response.text

def stream_answer(question:str)->None:
    """Stream the model response synchronously"""
    messages =[system_message,HumanMessage(content=question)]
    
    for chunk in model.astream(messages):
        print(chunk.text, end="",flush=True)
    print()

def ask_many(questions:list[str])->list[str]:
    """Process Independent questions concurrently"""
    inputs = [[system_message,HumanMessage(content=question)] for question in questions]

    responses= model.batch(
        inputs,
        config={
            "max_concurrency":3,
            "tags":["day02","batch"]
        }
    )
    return [response.text for response in responses]


async def ask_async(question:str)->str:
    """Invoke the model aynschronously"""
    messages = [system_message, HumanMessage(content=question)]
    response = await model.ainvoke(messages)
    return response.text

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a telecom billing assistant.

            You have no access to the customer's actual account.
            Never present possible causes as verified facts.
            """,
        ),
         (
            "human",
            """
            Customer question:
            {question}

            Explain:
            1. what can be answered now;
            2. what account information would be required;
            3. possible causes.
            """,
        ),
    ]
    
)


billing_chain= prompt | model

def ask_with_chain(question:str)-> str:
    """Invoke the composed model runnable"""
    
    response = billing_chain.invoke(
        {
            "question":question
        },
        config = {
            "run_name":"day02-billing-chain",
            "tags":["day02","runnable-chain"],
            "metadata":
                {
                    "prompt_version":"day02-v1"
                }
        }
    )
    return response.text 
async def main()->None:
    print("Synchronous invocation")
    print(ask("why might bill be $30 higher"))
    
if __name__=="__main__":
    asyncio.run(main())