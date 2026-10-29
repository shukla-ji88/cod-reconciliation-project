
from pathlib import Path
import pandas as pd

file_path = Path("data") / "Online Retail.xlsx"
df = pd.read_excel(file_path)

print("========== DATA QUALITY REPORT ==========")

# 1. Dataset size
print("\n1. Dataset shape:", df.shape)

# 2. Missing values
print("\n2. Missing values:")
print(df.isnull().sum())

# 3. Duplicate rows
print("\n3. Duplicate rows:", df.duplicated().sum())

# 4. Cancelled invoices
cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
print("\n4. Cancelled invoice rows:", cancelled.sum())

# 5. Negative quantities
print("\n5. Negative quantity rows:", (df["Quantity"] < 0).sum())

# 6. Invalid prices
print("\n6. Zero or negative price rows:", (df["UnitPrice"] <= 0).sum())

# 7. Transaction value
df["TransactionValue"] = df["Quantity"] * df["UnitPrice"]
print("\n7. Total transaction value:", round(df["TransactionValue"].sum(), 2))

print("\n========== REPORT COMPLETE ==========")
