
from pathlib import Path
import pandas as pd

# File paths
input_file = Path("data") / "Online Retail.xlsx"
output_file = Path("data") / "cleaned_online_retail.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Dataset not found: {input_file}"
    )

# Load original dataset
df = pd.read_excel(input_file)

print("Original rows:", len(df))

# 1. Remove exact duplicate rows only
duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicates)

# 2. Handle missing product descriptions
df["Description"] = df["Description"].fillna("Unknown")

# 3. Convert data types safely
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)
df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

# 4. Identify records that need special treatment
df["IsCancelled"] = (
    df["InvoiceNo"].astype(str).str.startswith("C")
)

df["IsReturnOrNegativeQty"] = df["Quantity"] < 0
df["IsInvalidPrice"] = df["UnitPrice"] < 0

# 5. Calculate transaction value
df["TransactionValue"] = (
    df["Quantity"] * df["UnitPrice"]
).round(2)

# 6. Export cleaned data
df.to_csv(output_file, index=False)

print("\nCleaned rows:", len(df))
print("Missing descriptions:", df["Description"].isna().sum())
print("Cancelled invoice rows:", df["IsCancelled"].sum())
print("Negative quantity rows:", df["IsReturnOrNegativeQty"].sum())
print("Negative price rows:", df["IsInvalidPrice"].sum())
print("Output file:", output_file)
print("File exists:", output_file.exists())
print("\nDATA CLEANING COMPLETED SUCCESSFULLY")
