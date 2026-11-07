
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
from visualization import suggest_visualization, create_ai_chart
from data_loader import load_dataset, build_report_context
from agent import create_data_analyst_agent, ask_datapilot
import os

# --------------------------------
# API Key Configuration
# --------------------------------

# Load the Groq API key from Streamlit Cloud Secrets
# if it is not already available locally through .env.

if not os.getenv("GROQ_API_KEY"):

    try:
        api_key = st.secrets["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = api_key

    except Exception:
        pass

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="DataPilot | AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("DataPilot")
st.caption("Your Autonomous AI Data Analyst")

st.write(
    "Upload a dataset, explore its patterns, "
    "visualize your data, and ask AI-powered questions."
)


# --------------------------------
# Session State Initialization
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "dataset_hash" not in st.session_state:
    st.session_state.dataset_hash = None

if "report" not in st.session_state:
    st.session_state.report = None


# --------------------------------
# Sidebar: Dataset Upload
# --------------------------------

st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to get started.")
    st.stop()


# --------------------------------
# Load Dataset
# --------------------------------

file_bytes = uploaded_file.getvalue()

current_hash = hashlib.sha256(file_bytes).hexdigest()

if current_hash != st.session_state.dataset_hash:

    try:
        uploaded_file.seek(0)
        df = load_dataset(uploaded_file)

        with st.spinner("Preparing your AI analyst..."):

            agent = create_data_analyst_agent(df)

        st.session_state.agent = agent
        st.session_state.dataset_hash = current_hash

        # Reset previous conversation and report
        st.session_state.messages = []
        st.session_state.report = None

    except Exception as error:
        st.error(f"Could not load the dataset: {error}")
        st.stop()

else:
    try:
        uploaded_file.seek(0)
        df = load_dataset(uploaded_file)

    except Exception as error:
        st.error(f"Could not load the dataset: {error}")
        st.stop()


st.sidebar.success("Dataset loaded successfully")


# --------------------------------
# Dataset Overview
# --------------------------------

st.divider()
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric("Duplicate Rows", int(df.duplicated().sum()))


# --------------------------------
# Dataset Preview
# --------------------------------

with st.expander("View Dataset Preview"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# --------------------------------
# Column Information
# --------------------------------

with st.expander("Column Information"):

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------
# Statistical Summary
# --------------------------------

with st.expander("Statistical Summary"):

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:

        st.dataframe(
            df[numeric_columns].describe().T,
            use_container_width=True
        )

    else:
        st.info("No numerical columns found.")


# --------------------------------
# Interactive Data Visualizations
# --------------------------------

st.divider()
st.header("Data Visualizations")

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

chart_type = st.selectbox(
    "Choose a visualization",
    [
        "Bar Chart",
        "Histogram",
        "Scatter Plot",
        "Box Plot",
        "Correlation Heatmap"
    ]
)


# Bar Chart
if chart_type == "Bar Chart":

    if numeric_columns and categorical_columns:

        category_column = st.selectbox(
            "Select category",
            categorical_columns,
            key="bar_category"
        )

        numeric_column = st.selectbox(
            "Select numerical column",
            numeric_columns,
            key="bar_numeric"
        )

        aggregation = st.selectbox(
            "Select aggregation",
            ["sum", "mean", "count"],
            key="bar_aggregation"
        )

        if aggregation == "count":

            chart_data = (
                df.groupby(category_column)
                .size()
                .reset_index(name="Count")
            )

            y_column = "Count"

        else:

            chart_data = (
                df.groupby(category_column)[numeric_column]
                .agg(aggregation)
                .reset_index()
            )

            y_column = numeric_column

        fig = px.bar(
            chart_data,
            x=category_column,
            y=y_column,
            title=f"{aggregation.title()} of {numeric_column} by {category_column}",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info(
            "A bar chart requires at least one categorical "
            "and one numerical column."
        )


# Histogram
elif chart_type == "Histogram":

    if numeric_columns:

        selected_column = st.selectbox(
            "Select numerical column",
            numeric_columns,
            key="histogram_column"
        )

        fig = px.histogram(
            df,
            x=selected_column,
            marginal="box",
            title=f"Distribution of {selected_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No numerical columns available.")


# Scatter Plot
elif chart_type == "Scatter Plot":

    if len(numeric_columns) >= 2:

        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox(
                "X-axis",
                numeric_columns,
                key="scatter_x"
            )

        with col2:
            y_column = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=1,
                key="scatter_y"
            )

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            trendline="ols",
            title=f"{x_column} vs {y_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info(
            "A scatter plot requires at least two numerical columns."
        )


# Box Plot
elif chart_type == "Box Plot":

    if numeric_columns:

        selected_column = st.selectbox(
            "Select numerical column",
            numeric_columns,
            key="box_column"
        )

        if categorical_columns:

            group_column = st.selectbox(
                "Group by (optional)",
                ["None"] + categorical_columns,
                key="box_group"
            )

            if group_column == "None":
                group_column = None

        else:
            group_column = None

        fig = px.box(
            df,
            x=group_column,
            y=selected_column,
            title=f"Box Plot of {selected_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No numerical columns available.")


# Correlation Heatmap
elif chart_type == "Correlation Heatmap":

    if len(numeric_columns) >= 2:

        correlation = df[numeric_columns].corr()

        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info(
            "A correlation heatmap requires at least two numerical columns."
        )


# --------------------------------
# AI-Powered Autonomous Visualization
# --------------------------------

st.divider()
st.header("AI-Powered Visualization")

st.write(
    "Describe the chart you want in natural language. "
    "DataPilot will choose the chart type and configure it automatically."
)

visualization_question = st.text_input(
    "What would you like to visualize?",
    placeholder="Example: Compare total sales across regions."
)

if st.button("Generate AI Visualization"):

    if not visualization_question.strip():

        st.warning("Please describe the visualization you want.")

    else:

        with st.spinner("DataPilot is selecting the best visualization..."):

            try:

                # Ask AI to select the chart configuration
                chart_config = suggest_visualization(
                    df,
                    visualization_question,
                    st.session_state.messages
                )

                # Validate configuration and generate chart
                fig = create_ai_chart(
                    df,
                    chart_config
                )

                st.success("Visualization generated successfully!")

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                with st.expander("View AI Chart Configuration"):
                    st.json(chart_config)

            except Exception as error:

                st.error(
                    f"Visualization generation failed: {error}"
                )


# --------------------------------
# AI Analyst Chat
# --------------------------------

st.divider()
st.header("AI Analyst Chat")

st.write(
    "Ask questions about your dataset. "
    "DataPilot remembers previous questions and answers."
)

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
question = st.chat_input(
    "Ask DataPilot a question about your data..."
)

if question:

    # Save the new question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your data..."):

            try:

                # Pass all previous messages except current question
                chat_history = st.session_state.messages[:-1]

                answer = ask_datapilot(
                    st.session_state.agent,
                    question,
                    chat_history
                )

                st.markdown(answer)

                # Save AI response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:

                st.error(f"Analysis failed: {error}")

# --------------------------------
# AI-Generated Analytical Report
# --------------------------------

st.divider()
st.header("AI Analytical Report")

st.write(
    "Generate a structured analytical report containing "
    "statistical findings, patterns, and recommendations."
)


if st.button(
    "Generate Analytical Report",
    type="primary"
):

    report_question = """
    Prepare a complete professional analytical report
    using the Python-calculated analysis results.

    Use these sections:

    # Executive Summary
    # Dataset Overview
    # Key Findings
    # Statistical Analysis
    # Recommendations
    # Limitations

    Requirements:

    1. Use the supplied Python results as the source of truth.
    2. Do not invent or recalculate numerical statistics.
    3. Distinguish total values from average values.
    4. Identify the most important patterns.
    5. Explain correlations without implying causation.
    6. Discuss outliers where applicable.
    7. Avoid strong recommendations based on very small samples.
    8. Distinguish observed facts from possible explanations.
    9. Mention limitations, including small sample sizes.
    10. Complete every report section.
    11. Use clear professional language.
    """

    with st.spinner(
        "Calculating statistics and preparing your report..."
    ):

        try:

            # Step 1: Calculate statistics using Python
            report_context = build_report_context(df)

            # Step 2: Give the verified results to the AI
            full_question = f"""
            You are DataPilot, an AI data analyst.

            The following results were calculated directly
            using Python and Pandas.

            Treat them as the source of truth.

            PYTHON ANALYSIS RESULTS:
            {report_context}

            USER REQUEST:
            {report_question}

            Do not contradict the supplied numerical results.
            """

            # Step 3: Generate the report
            report = ask_datapilot(
                st.session_state.agent,
                full_question,
            )

            st.session_state.report = report

        except Exception as error:

            st.error(
                f"Report generation failed: {error}"
            )


# --------------------------------
# Display and Download Report
# --------------------------------

if st.session_state.report:

    st.subheader("Generated Report")

    st.markdown(
        st.session_state.report
    )

    report_content = f"""# DataPilot Analytical Report

**Dataset:** {uploaded_file.name}

**Rows:** {len(df)}

**Columns:** {len(df.columns)}

**Generated:** {pd.Timestamp.now()}

---

{st.session_state.report}
"""

    st.download_button(
        label="Download Report",
        data=report_content,
        file_name="DataPilot_Analytical_Report.md",
        mime="text/markdown"
    )
