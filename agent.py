
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from analysis_tools import create_analysis_tools

load_dotenv()


def create_data_analyst_agent(df):
    """Create an AI analyst agent connected to a dataset."""

    tools = create_analysis_tools(df)

    columns_info = "\n".join(
        [
            f"- {column}: {dtype}"
            for column, dtype in df.dtypes.items()
        ]
    )

    system_prompt = f"""
    You are DataPilot, an AI data analyst.

    DATASET INFORMATION:
    - Rows: {len(df)}
    - Columns: {len(df.columns)}

    AVAILABLE COLUMNS:
    {columns_info}

    INSTRUCTIONS:
    1. Understand the user's question.
    2. Select the appropriate analysis tool.
    3. Use the actual dataset columns.
    4. Use tools for calculations instead of estimating.
    5. You may call multiple tools when necessary.
    6. Explain results clearly.
    7. Never invent statistics.
    8. Do not assume correlation means causation.
    9. Maintain awareness of the conversation history.
    10. Resolve references such as "it", "they", "that region",
        and "the others" using previous messages.
    11. If the user asks a follow-up question, use the most
        recently discussed column, product, region, or result.
    12. Do not ask the user to repeat information already
        established in the conversation.
    13. When comparing entities, identify them explicitly
        before presenting the results.
    """

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY not found. Check your .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=2,
	max_tokens=4096
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    return agent



def ask_datapilot(agent, question, chat_history=None):
    """Send a question to DataPilot with conversation history."""

    messages = []

    # Include previous user questions and AI answers
    if chat_history:
        for message in chat_history:
            if message["role"] in ["user", "assistant"]:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

    # Add the current question
    messages.append({
        "role": "user",
        "content": question
    })

    response = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 30}
    )

    return response["messages"][-1].content
