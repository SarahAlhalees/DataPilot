
from data_loader import load_dataset
from analysis_tools import create_analysis_tools

# Load our sample dataset
df = load_dataset("sample_sales.csv")

# Create the analysis tools
tools = create_analysis_tools(df)

# Test each tool
print("\n1. SALES SUMMARY")
print(tools[0].invoke({"column": "Sales"}))

print("\n2. SALES BY REGION")
print(tools[1].invoke({
    "category_column": "Region",
    "numeric_column": "Sales",
    "operation": "sum"
}))

print("\n3. SALES AND QUANTITY CORRELATION")
print(tools[2].invoke({
    "column1": "Sales",
    "column2": "Quantity"
}))

print("\n4. SALES OUTLIERS")
print(tools[3].invoke({"column": "Sales"}))

print("\n5. TOP PRODUCTS")
print(tools[4].invoke({
    "column": "Product",
    "limit": 5
}))
