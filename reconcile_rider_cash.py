
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data") / "simulated_cod"

collections_file = DATA_DIR / "collection_transactions.csv"
handovers_file = DATA_DIR / "rider_cash_handovers.csv"

# 1. Check input files
for file in [collections_file, handovers_file]:
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

# 2. Load data
collections = pd.read_csv(collections_file)
handovers = pd.read_csv(handovers_file)

# 3. Validate required fields
required_collections = {
    "collection_id",
    "rider_id",
    "payment_mode",
    "payment_status",
    "collected_amount",
    "transaction_reference",
    "collection_time",
}

required_handovers = {
    "handover_id",
    "rider_id",
    "handover_date",
    "deposited_cash_amount",
}

if not required_collections.issubset(collections.columns):
    raise ValueError("Required collection columns are missing")

if not required_handovers.issubset(handovers.columns):
    raise ValueError("Required handover columns are missing")

# 4. Normalize values and types
collections["rider_id"] = collections["rider_id"].astype(str).str.strip()
collections["payment_mode"] = (
    collections["payment_mode"].astype(str).str.upper().str.strip()
)
collections["payment_status"] = (
    collections["payment_status"].astype(str).str.upper().str.strip()
)
collections["collected_amount"] = pd.to_numeric(
    collections["collected_amount"], errors="coerce"
)
collections["collection_time"] = pd.to_datetime(
    collections["collection_time"], errors="coerce"
)
collections["transaction_reference"] = (
    collections["transaction_reference"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
)

handovers["rider_id"] = handovers["rider_id"].astype(str).str.strip()
handovers["deposited_cash_amount"] = pd.to_numeric(
    handovers["deposited_cash_amount"], errors="coerce"
)
handovers["handover_date"] = pd.to_datetime(
    handovers["handover_date"], errors="coerce"
)

if collections["collected_amount"].isna().any():
    raise ValueError("Invalid collection amount found")

if collections["collection_time"].isna().any():
    raise ValueError("Invalid collection timestamp found")

if handovers["deposited_cash_amount"].isna().any():
    raise ValueError("Invalid cash deposit amount found")

if handovers["handover_date"].isna().any():
    raise ValueError("Invalid handover date found")

if (handovers["deposited_cash_amount"] < 0).any():
    raise ValueError("Negative cash deposit found; review source data")

# 5. Remove additional records with a repeated transaction reference
# This prevents duplicated collection rows from inflating cash totals.
has_reference = collections["transaction_reference"].notna()

extra_duplicate = (
    has_reference
    & collections["transaction_reference"].duplicated(keep="first")
)

duplicate_rows = collections.loc[extra_duplicate].copy()
clean_collections = collections.loc[~extra_duplicate].copy()

# 6. Select successful cash collections only
cash = clean_collections.loc[
    clean_collections["payment_mode"].eq("CASH")
    & clean_collections["payment_status"].eq("SUCCESS")
].copy()

cash["business_date"] = cash["collection_time"].dt.normalize()

# 7. Calculate expected cash per rider per day
expected = (
    cash.groupby(["rider_id", "business_date"], as_index=False)
    .agg(
        successful_cash_transactions=("collection_id", "count"),
        expected_cash_amount=("collected_amount", "sum")
    )
)

# 8. Calculate actual deposits per rider per day
handovers["business_date"] = handovers["handover_date"].dt.normalize()

actual = (
    handovers.groupby(["rider_id", "business_date"], as_index=False)
    .agg(
        handover_record_count=("handover_id", "count"),
        deposited_cash_amount=("deposited_cash_amount", "sum")
    )
)

# 9. Merge both sides, keeping missing handovers and unmatched deposits
report = expected.merge(
    actual,
    on=["rider_id", "business_date"],
    how="outer",
    validate="one_to_one"
)

for column in [
    "successful_cash_transactions",
    "handover_record_count"
]:
    report[column] = report[column].fillna(0).astype(int)

for column in ["expected_cash_amount", "deposited_cash_amount"]:
    report[column] = report[column].fillna(0.0)

report["cash_gap"] = (
    report["expected_cash_amount"]
    - report["deposited_cash_amount"]
).round(2)

# 10. Assign review statuses
def classify_handover(row):
    expected_cash = row["expected_cash_amount"]
    deposited = row["deposited_cash_amount"]
    gap = row["cash_gap"]

    if expected_cash > 0 and row["handover_record_count"] == 0:
        return "MISSING_HANDOVER"

    if abs(gap) <= 0.01:
        return "MATCHED"

    if gap > 0:
        return "SHORT_DEPOSIT"

    return "OVER_DEPOSIT"

report["handover_status"] = report.apply(
    classify_handover, axis=1
)

# 11. Save reports
report_path = DATA_DIR / "rider_cash_reconciliation.csv"
duplicate_path = DATA_DIR / "rider_cash_duplicate_transactions.csv"

report.to_csv(report_path, index=False)
duplicate_rows.to_csv(duplicate_path, index=False)

# 12. Print summary
print("=" * 55)
print("RIDER CASH HANDOVER RECONCILIATION")
print("=" * 55)

print("\nUnique rider-date records:", len(report))
print("Duplicate collection records excluded:", len(duplicate_rows))

print("\nHandover status:")
print(report["handover_status"].value_counts().to_string())

print(
    "\nExpected successful cash: ₹",
    f"{report['expected_cash_amount'].sum():,.2f}"
)
print(
    "Recorded cash deposits: ₹",
    f"{report['deposited_cash_amount'].sum():,.2f}"
)
print(
    "Net cash gap: ₹",
    f"{report['cash_gap'].sum():,.2f}"
)

print("\nSaved:", report_path)
print("Saved:", duplicate_path)
print("\nRIDER CASH RECONCILIATION COMPLETED SUCCESSFULLY")
