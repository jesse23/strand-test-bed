from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator
import os
from dotenv import load_dotenv
import base64
import PyPDF2

# Load environment variables from .env file
load_dotenv()

model = OpenAIModel(
    client_args={
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    model_id="gpt-4",  # Using GPT-4 for better PDF analysis
    params={
        # "max_tokens": 2000,
        "temperature": 0.7,
    }
)

def extract_text_from_pdf(file_path, max_pages=5):
    """Extract text from PDF, limiting to first N pages to manage size."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        # Only process first N pages
        for i in range(min(max_pages, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
    return text

# Create the agent with calculator tool
agent = Agent(model=model, tools=[calculator])

# Example usage with PDF
pdf_path = "./0_knowledge/test.pdf"  # Replace with your PDF path
pdf_text = extract_text_from_pdf(pdf_path)

prompt = f"""
Please analyze this PDF content and answer the following questions:
1. What are the main topics discussed in the document?
2. What are the key findings or conclusions?
3. What is 2+2? (use the calculator tool for this)

PDF content:
{pdf_text}
"""

response = agent(prompt)
print(response)
