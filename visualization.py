
import json
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def suggest_visualization(df, question, chat_history=None):
    """Ask the LLM to select an appropriate chart."""

    columns = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    sample = df.head(3).to_dict(orient="records")

    history = ""

    if chat_history:
        history = "\n".join(
            f'{m["role"]}: {m["content"]}'
            for m in chat_history[-6:]
        )

    prompt = f"""
    You are a data visualization specialist.

    Choose the most appropriate chart for the user's request.

    DATASET COLUMNS:
    {json.dumps(columns)}

    SAMPLE DATA:
    {json.dumps(sample, default=str)}

    CONVERSATION HISTORY:
    {history}

    USER REQUEST:
    {question}

    Return ONLY a valid JSON object with these fields:

    {{
        "chart_type": "bar",
        "x": "Region",
        "y": "Sales",
        "aggregation": "sum",
        "title": "Total Sales by Region"
    }}

    RULES:
    - Allowed chart types: bar, histogram, scatter, box, line.
    - Use actual column names.
    - Bar charts can use sum, mean, or count.
    - Histogram and box charts need a numerical column.
    - Scatter charts need two numerical columns.
    - Line charts need an appropriate x and numerical y.
    - Use null when a field is not needed.
    - Do not invent columns.
    """

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=2
    )

    response = llm.invoke(prompt)

    result = json.loads(response.content)

    return result


def create_ai_chart(df, config):
    """Validate the AI chart configuration and create a Plotly chart."""

    chart_type = config.get("chart_type")
    x = config.get("x")
    y = config.get("y")
    aggregation = config.get("aggregation")
    title = config.get("title", "AI-Generated Visualization")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if x not in df.columns and x is not None:
        raise ValueError("Invalid X-axis column.")

    if y not in df.columns and y is not None:
        raise ValueError("Invalid Y-axis column.")

    if chart_type == "bar":

        if x is None:
            raise ValueError("Bar chart requires an X-axis column.")

        if aggregation == "count":

            chart_data = (
                df.groupby(x)
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                chart_data,
                x=x,
                y="Count",
                title=title
            )

        else:

            if y not in numeric_columns:
                raise ValueError("Bar chart requires a numerical Y-axis.")

            if aggregation not in ["sum", "mean"]:
                raise ValueError("Invalid bar chart aggregation.")

            chart_data = (
                df.groupby(x)[y]
                .agg(aggregation)
                .reset_index()
            )

            fig = px.bar(
                chart_data,
                x=x,
                y=y,
                title=title
            )

    elif chart_type == "histogram":

        if x not in numeric_columns:
            raise ValueError("Histogram requires a numerical column.")

        fig = px.histogram(
            df,
            x=x,
            title=title
        )

    elif chart_type == "scatter":

        if x not in numeric_columns or y not in numeric_columns:
            raise ValueError("Scatter plot requires two numerical columns.")

        fig = px.scatter(
            df,
            x=x,
            y=y,
            title=title
        )

    elif chart_type == "box":

        if y not in numeric_columns:
            raise ValueError("Box plot requires a numerical Y-axis.")

        fig = px.box(
            df,
            x=x,
            y=y,
            title=title
        )

    elif chart_type == "line":

        if y not in numeric_columns:
            raise ValueError("Line chart requires a numerical Y-axis.")

        fig = px.line(
            df,
            x=x,
            y=y,
            title=title
        )

    else:
        raise ValueError("Unsupported chart type.")

    return fig
