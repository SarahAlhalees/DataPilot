import pandas as pd
from data_loader import load_dataset, profile_dataset

# Create a sample dataset
data = {
    "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone"],
    "Sales": [5000, 3000, 2000, 4500, 3500],
    "Quantity": [5, 10, 4, 4, 12],
    "Region": ["Jeddah", "Riyadh", "Jeddah", "Dammam", "Riyadh"]
}

df = pd.DataFrame(data)

# Save as CSV
df.to_csv("sample_sales.csv", index=False)

# Load the dataset
loaded_df = load_dataset("sample_sales.csv")

# Generate dataset profile
profile = profile_dataset(loaded_df)

print("DATASET PROFILE")
print("--------------------")

for key, value in profile.items():
    print(f"\n{key}:")
    print(value)