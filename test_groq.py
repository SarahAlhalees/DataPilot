import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY was not found. Check your .env file.")

# Connect to Groq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2
)

# Send a test message
response = llm.invoke("Reply with: DataPilot connection successful!")

print(response.content)