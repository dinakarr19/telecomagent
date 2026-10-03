import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from pathlib import Path
env_path=Path(__file__).resolve().parents[2]/".env"

load_dotenv(env_path)
model_id=os.getenv("MODEL_ID")

if not model_id:
    raise RuntimeError("Model id is not configured")
model=init_chat_model(model_id, temperature=0,timeout=30)

system_message = SystemMessage("""You are a telecom billing assistant.

You currently have no access to the customer's real account.

Never invent billing details.

When information is unavailable:
1. state that limitation;
2. identify what information would be required;
3. explain possible causes without presenting them as facts.""")

user_message = HumanMessage("Why is my bill $30 higher?")



response = model.invoke([system_message,user_message])
print(response.text)
