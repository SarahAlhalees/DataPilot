
import json
import pandas as pd
from langchain_core.tools import tool


def create_analysis_tools(df):
    """Create analysis tools connected to a specific dataset."""

    @tool
    def summarize_column(column: str) -> str:
        """Summarize a dataset column, including statistics or value frequencies."""

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            result = {
                "column": column,
                "type": "numeric",
                "count": int(series.count()),
                "mean": float(series.mean()) if series.count() else None,
                "median": float(series.median()) if series.count() else None,
                "minimum": float(series.min()) if series.count() else None,
                "maximum": float(series.max()) if series.count() else None,
                "standard_deviation": float(series.std()) if series.count() > 1 else None
            }
        else:
            result = {
                "column": column,
                "type": "categorical",
                "unique_values": int(series.nunique()),
                "most_common": series.value_counts().head(10).to_dict()
            }

        return json.dumps(result, default=str)

    @tool
    def compare_groups(
        category_column: str,
        numeric_column: str,
        operation: str = "mean"
    ) -> str:
        """Compare a numerical metric across categories. Operations: mean, sum, median, count."""

        if category_column not in df.columns:
            return f"Column '{category_column}' does not exist."

        if numeric_column not in df.columns:
            return f"Column '{numeric_column}' does not exist."

        if not pd.api.types.is_numeric_dtype(df[numeric_column]):
            return f"'{numeric_column}' must be numeric."

        if operation not in ["mean", "sum", "median", "count"]:
            return "Invalid operation. Use mean, sum, median, or count."

        grouped = df.groupby(category_column)[numeric_column]

        result = getattr(grouped, operation)().sort_values(
            ascending=False
        ).to_dict()

        return json.dumps(result, default=str)

    @tool
    def calculate_correlation(column1: str, column2: str) -> str:
        """Calculate the Pearson correlation between two numerical columns."""

        for column in [column1, column2]:
            if column not in df.columns:
                return f"Column '{column}' does not exist."

            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"'{column}' must be numeric."

        correlation = df[column1].corr(df[column2])

        if pd.isna(correlation):
            return "Correlation could not be calculated."

        return json.dumps({
            "column1": column1,
            "column2": column2,
            "correlation": round(float(correlation), 4)
        })

    @tool
    def detect_outliers(column: str) -> str:
        """Detect outliers in a numerical column using the IQR method."""

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"'{column}' must be numeric."

        series = df[column].dropna()

        if len(series) < 4:
            return "Not enough values to reliably detect outliers."

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ][[column]]

        result = {
            "column": column,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": len(outliers),
            "outliers": outliers.to_dict(orient="records")
        }

        return json.dumps(result, default=str)

    @tool
    def get_top_values(column: str, limit: int = 5) -> str:
        """Find the most frequent values in a dataset column."""

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        limit = max(1, min(limit, 20))

        result = df[column].value_counts().head(limit).to_dict()

        return json.dumps(result, default=str)

    return [
        summarize_column,
        compare_groups,
        calculate_correlation,
        detect_outliers,
        get_top_values
    ]
