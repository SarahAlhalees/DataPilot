import pandas as pd


def load_dataset(file):
    """Load a CSV or Excel dataset."""

    file_name = file.name if hasattr(file, "name") else str(file)
    extension = file_name.split(".")[-1].lower()

    if extension == "csv":
        df = pd.read_csv(file)

    elif extension in ["xlsx", "xls"]:
        df = pd.read_excel(file)

    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    return df


def profile_dataset(df):
    """Generate a basic profile of the dataset."""

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "preview": df.head(5).to_dict(orient="records"),
        "numeric_summary": df.describe().to_dict()
    }

    return profile

def build_report_context(df):
    """Calculate reliable statistics for the AI report."""

    import pandas as pd

    results = []

    results.append(f"Rows: {len(df)}")
    results.append(f"Columns: {len(df.columns)}")

    # Numerical analysis
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        results.append(f"\nStatistics for {column}:")
        results.append(str(df[column].describe().round(2)))

        # Outlier detection using IQR
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[
            (df[column] < lower) |
            (df[column] > upper)
        ]

        results.append(
            f"Outliers: {len(outliers)}"
        )

    # Categorical analysis
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        results.append(f"\nDistribution of {column}:")
        results.append(str(df[column].value_counts()))

    # Group comparisons
    for category in categorical_columns:
        for numeric in numeric_columns:
            grouped = df.groupby(category)[numeric].agg(
                ["sum", "mean", "count"]
            ).round(2)

            results.append(
                f"\n{numeric} grouped by {category}:"
            )
            results.append(str(grouped))

    # Correlations
    if len(numeric_columns) > 1:
        results.append("\nCorrelation matrix:")
        results.append(
            str(df[numeric_columns].corr().round(3))
        )

    return "\n".join(results)