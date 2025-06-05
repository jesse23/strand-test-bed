from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator
import os
from dotenv import load_dotenv
import PyPDF2
import tiktoken

# Load environment variables from .env file
load_dotenv()

model = OpenAIModel(
    client_args={
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    model_id="gpt-4",  # Using GPT-4 for better PDF analysis
    params={
        "max_tokens": 2000,
        "temperature": 0.7,
    }
)

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))

def extract_text_from_pdf(file_path: str, max_tokens_per_chunk: int = 1500) -> list:
    """
    Extract text from PDF and split into chunks that fit within token limits.
    Returns a list of text chunks.
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        chunks = []
        current_chunk = ""
        
        for page in reader.pages:
            page_text = page.extract_text() + "\n"
            # If adding this page would exceed the token limit, save current chunk and start new one
            if count_tokens(current_chunk + page_text) > max_tokens_per_chunk:
                if current_chunk:  # Don't append empty chunks
                    chunks.append(current_chunk)
                current_chunk = page_text
            else:
                current_chunk += page_text
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
    return chunks

def analyze_chunks(agent, chunks: list) -> str:
    """Analyze each chunk and combine the results."""
    all_analyses = []
    
    for i, chunk in enumerate(chunks, 1):
        prompt = f"""
        Please analyze this section of the PDF (part {i} of {len(chunks)}) and answer:
        1. What are the main topics in this section?
        2. What are the key points or findings?

        Content:
        {chunk}
        """
        
        response = agent(prompt)
        all_analyses.append(f"Analysis of Section {i}:\n{response}\n")
    
    # Get overall analysis
    summary_prompt = f"""
    Based on the following analyses of different sections of the document, provide a comprehensive summary:
    
    {'\n'.join(all_analyses)}
    
    Please provide:
    1. Overall main topics
    2. Key findings across all sections
    3. How do the different sections relate to each other?
    """
    
    final_summary = agent(summary_prompt)
    return final_summary

# Create the agent with calculator tool
agent = Agent(model=model, tools=[calculator])

# Example usage with PDF
pdf_path = "./0_knowledge/test.pdf"  # Replace with your PDF path
chunks = extract_text_from_pdf(pdf_path)

# Analyze all chunks and get final summary
final_analysis = analyze_chunks(agent, chunks)
print(final_analysis) 