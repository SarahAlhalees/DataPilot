# DataPilot — Autonomous AI Data Analyst

**DataPilot** is an AI-powered data analysis application that allows users to explore datasets, generate visualizations, and obtain analytical insights using natural language.

Built with Python, LangGraph, and Groq, DataPilot combines an AI agent with Python-based analytical tools to perform data analysis and explain results in an accessible way.

## Features

### 1. Dataset Upload & Exploration
- Upload CSV and Excel files.
- Preview datasets and inspect column information.
- View dataset dimensions, data types, missing values, and duplicate records.
- Generate descriptive statistical summaries.

### 2. AI-Powered Data Analysis
- Ask questions about datasets using natural language.
- Use a LangGraph-based AI agent to understand requests and select appropriate analytical tools.
- Perform statistical analysis using Pandas and NumPy.
- Compare categories, calculate correlations, identify outliers, and summarize columns.
- Maintain conversation context for follow-up questions.

### 3. Autonomous Data Visualization
- Generate charts from natural language requests.
- Automatically select suitable chart types and dataset columns.
- Support bar charts, histograms, scatter plots, box plots, and line charts.
- Use Plotly for interactive visualizations.

### 4. Automated Analytical Reports
- Generate structured analytical reports.
- Use Python-calculated statistics as the source of numerical findings.
- Identify patterns, summarize findings, and provide data-driven recommendations.
- Download reports in Markdown format.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| AI Agent Framework | LangGraph |
| LLM Integration | LangChain |
| LLM Provider | Groq Cloud |
| Language Model | GPT-OSS-20B |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Statistical Analysis | Statsmodels |
| User Interface | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

## System Architecture

1. **Data Input:** Users upload CSV or Excel datasets through the Streamlit interface.
2. **Data Processing:** Python processes the dataset and generates statistical information.
3. **AI Agent:** LangGraph manages the agent workflow and tool selection.
4. **Tool Execution:** Python analytical tools perform calculations on the uploaded data.
5. **AI Interpretation:** Groq-powered LLM interprets the results and generates explanations.
6. **Visualization:** The AI selects appropriate chart configurations, which are generated using Plotly.
7. **Output:** Users receive analytical answers, interactive charts, and downloadable reports.

## Project Structure

```text
DataPilot/
│
├── app.py
├── agent.py
├── analysis_tools.py
├── data_loader.py
├── visualization.py
│
├── requirements.txt
├── .gitignore
├── sample_sales.csv
│
├── test_agent.py
├── test_analysis_tools.py
├── test_groq.py
└── test_loader.py
```

## Getting Started

### Prerequisites
- Python 3.12 or later
- A Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/SarahAlhalees/DataPilot.git
cd DataPilot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

You can obtain an API key from [Groq Console](https://console.groq.com/).

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Deployment

DataPilot is deployed using **Streamlit Community Cloud**.

The application uses:
- GitHub for source code management.
- Streamlit Community Cloud for application hosting.
- Groq Cloud for LLM inference.

For cloud deployment, configure `GROQ_API_KEY` in the application's Streamlit Secrets settings.

## Future Improvements

- Support additional file formats and larger datasets.
- Expand the analytical toolkit with advanced statistical methods.
- Improve agent planning and multi-step reasoning.
- Add data cleaning and preprocessing recommendations.
- Introduce predictive analytics and machine learning capabilities.
- Support exporting reports in PDF and Excel formats.

## Live Application

**Try DataPilot:** [Open DataPilot](https://datapilot.streamlit.app/)

*Note: Replace this URL with your actual Streamlit application URL if its generated address is different.*

## Author

**Sarah Alhalees**

AI-focused developer interested in Artificial Intelligence, Machine Learning, Natural Language Processing, and building practical AI-powered applications.

---

*DataPilot — Turn your data into insights through natural language.*
