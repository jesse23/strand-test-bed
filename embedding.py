from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import file_read, file_write, editor
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
        # "max_tokens": 1000,
        "temperature": 0.7,
    }
)

# Define a focused system prompt for file operations
FILE_SYSTEM_PROMPT = """Summarize the content of the PDF file. If OCR is needed please do it.
"""

# Create a file-focused agent with selected tools
file_agent = Agent(
    model=model,
    system_prompt=FILE_SYSTEM_PROMPT,
    tools=[],
)

resp = file_agent("Read ./test.pdf")
print(resp)

