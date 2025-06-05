from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model = OpenAIModel(
    client_args={
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    # **model_config
    model_id="gpt-4o-mini",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)