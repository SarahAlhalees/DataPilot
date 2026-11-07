
from data_loader import load_dataset
from agent import create_data_analyst_agent, ask_datapilot

# Load the sample dataset
df = load_dataset("sample_sales.csv")

# Create DataPilot
agent = create_data_analyst_agent(df)

print("\nDataPilot is ready!")
print("Ask a question about your dataset.\n")

# Test the agent
question = "Which region generated the highest total sales?"

answer = ask_datapilot(agent, question)

print("QUESTION:")
print(question)

print("\nDATAPILOT ANSWER:")
print(answer)
