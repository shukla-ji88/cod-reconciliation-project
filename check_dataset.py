
from pathlib import Path
import pandas as pd

file_path = Path("data") / "Online Retail.xlsx"

if not file_path.exists():
    print("Dataset not found. Check the data folder and filename.")
else:
    df = pd.read_excel(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
