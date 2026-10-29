
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data") / "simulated_cod"

orders_file = DATA_DIR / "orders.csv"
collections_file = DATA_DIR / "collection_transactions.csv"

# 1. Verify input files
for file in [orders_file, collections_file]:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found: {file}. "
            "Run generate_cod_data.py first."
        )

# 2. Load data
orders = pd.read_csv(orders_file)
collections = pd.read_csv(collections_file)

# 3. Normalize important fields
orders["order_id"] = orders["order_id"].astype(str).str.strip()
orders["order_status"] = (
    orders["order_status"].astype(str).str.upper().str.strip()
)
orders["cod_amount"] = pd.to_numeric(
    orders["cod_amount"], errors="coerce"
)

collections["order_id"] = (
    collections["order_id"].astype(str).str.strip()
)
collections["payment_status"] = (
    collections["payment_status"].astype(str).str.upper().str.strip()
)
collections["payment_mode"] = (
    collections["payment_mode"].astype(str).str.upper().str.strip()
)
collections["collected_amount"] = pd.to_numeric(
    collections["collected_amount"], errors="coerce"
)

# Normalize transaction references
collections["transaction_reference"] = (
    collections["transaction_reference"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
)

# 4. Validate orders
if orders["order_id"].duplicated().any():
    raise ValueError("Duplicate order IDs found in orders.csv")

if orders["cod_amount"].isna().any():
    raise ValueError("Orders contain missing or invalid COD amounts")

if collections["collected_amount"].isna().any():
    raise ValueError(
        "Collections contain missing or invalid collection amounts"
    )

# 5. Identify every row participating in a duplicate reference
valid_ref = collections["transaction_reference"].notna()

duplicate_group_mask = (
    valid_ref
    & collections["transaction_reference"].duplicated(keep=False)
)

duplicate_report = collections.loc[duplicate_group_mask].copy()

# Count only the additional occurrences, not the original transaction.
extra_duplicate_mask = (
    valid_ref
    & collections["transaction_reference"].duplicated(keep="first")
)

extra_duplicates = collections.loc[extra_duplicate_mask].copy()

# 6. Exclude repeated references from financial totals.
# Keep one record per non-empty transaction reference.
# Records with no reference are retained but cannot be deduplicated by ref.
unique_collections = collections.loc[
    ~extra_duplicate_mask
].copy()

# 7. Build collection metrics by order
unique_collections["successful_amount"] = np.where(
    unique_collections["payment_status"].eq("SUCCESS"),
    unique_collections["collected_amount"],
    0.0
)

unique_collections["is_success"] = (
    unique_collections["payment_status"].eq("SUCCESS").astype(int)
)
unique_collections["is_pending"] = (
    unique_collections["payment_status"].eq("PENDING").astype(int)
)
unique_collections["is_failed"] = (
    unique_collections["payment_status"].eq("FAILED").astype(int)
)

metrics = (
    unique_collections.groupby("order_id", as_index=False)
    .agg(
        collection_record_count=("collection_id", "count"),
        successful_transaction_count=("is_success", "sum"),
        pending_transaction_count=("is_pending", "sum"),
        failed_transaction_count=("is_failed", "sum"),
        successful_collected_amount=("successful_amount", "sum")
    )
)

# Count additional duplicate references per order
duplicate_counts = (
    extra_duplicates.groupby("order_id")
    .size()
    .rename("duplicate_reference_count")
    .reset_index()
)

# 8. Merge orders with collection metrics
report = orders.merge(
    metrics, on="order_id", how="left", validate="one_to_one"
)

report = report.merge(
    duplicate_counts, on="order_id", how="left", validate="one_to_one"
)

count_columns = [
    "collection_record_count",
    "successful_transaction_count",
    "pending_transaction_count",
    "failed_transaction_count",
    "duplicate_reference_count"
]

for column in count_columns:
    report[column] = report[column].fillna(0).astype(int)

report["successful_collected_amount"] = (
    report["successful_collected_amount"].fillna(0).round(2)
)

# 9. Only delivered orders are eligible for the expected COD amount.
delivered = report["order_status"].eq("DELIVERED")

report["expected_cod_amount"] = np.where(
    delivered,
    report["cod_amount"],
    0.0
)

report["collection_gap"] = (
    report["expected_cod_amount"]
    - report["successful_collected_amount"]
).round(2)

# 10. Assign reconciliation status
def classify_order(row):
    is_delivered = row["order_status"] == "DELIVERED"
    records = row["collection_record_count"]
    successful = row["successful_transaction_count"]
    pending = row["pending_transaction_count"]
    failed = row["failed_transaction_count"]
    duplicates = row["duplicate_reference_count"]
    gap = row["collection_gap"]

    if not is_delivered:
        if successful > 0:
            return "COLLECTION_ON_NON_DELIVERED_ORDER"
        if records > 0:
            return "NON_DELIVERED_ORDER_WITH_PAYMENT_ATTEMPT"
        return "NOT_ELIGIBLE_NON_DELIVERED"

    if successful == 0:
        if pending > 0:
            return "PAYMENT_PENDING"
        if failed > 0:
            return "PAYMENT_FAILED"
        return "MISSING_COLLECTION"

    if abs(gap) > 0.01:
        return "AMOUNT_MISMATCH"

    if duplicates > 0:
        return "MATCHED_WITH_DUPLICATE_REFERENCE"

    return "MATCHED"


report["reconciliation_status"] = report.apply(
    classify_order, axis=1
)

# 11. Detect collection records linked to unknown orders
unknown_order_collections = collections.loc[
    ~collections["order_id"].isin(orders["order_id"])
].copy()

# 12. Save reports
report_path = DATA_DIR / "reconciliation_report.csv"
duplicates_path = DATA_DIR / "duplicate_transactions.csv"
unknown_path = DATA_DIR / "unknown_order_collections.csv"

report.to_csv(report_path, index=False)
duplicate_report.to_csv(duplicates_path, index=False)
unknown_order_collections.to_csv(unknown_path, index=False)

# 13. Print summary
print("=" * 52)
print("COD RECONCILIATION REPORT")
print("=" * 52)

print("\nOrders processed:", len(report))
print("Raw collection records:", len(collections))
print("Additional duplicate references:", len(extra_duplicates))
print("Rows in duplicate reference report:", len(duplicate_report))
print("Collections linked to unknown orders:", len(unknown_order_collections))

print("\nReconciliation status:")
print(report["reconciliation_status"].value_counts().to_string())

print(
    "\nExpected COD for delivered orders: ₹",
    f"{report.loc[delivered, 'expected_cod_amount'].sum():,.2f}"
)

print(
    "Successful recorded collections on delivered orders: ₹",
    f"{report.loc[delivered, 'successful_collected_amount'].sum():,.2f}"
)

print(
    "Net collection gap on delivered orders: ₹",
    f"{report.loc[delivered, 'collection_gap'].sum():,.2f}"
)

print("\nSaved report:", report_path)
print("Saved duplicate report:", duplicates_path)
print("Saved unknown-order report:", unknown_path)
print("\nCOD RECONCILIATION COMPLETED SUCCESSFULLY")
