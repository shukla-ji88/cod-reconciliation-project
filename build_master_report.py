
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data") / "simulated_cod"

def load_report(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing report: {path}. Run the previous reconciliation scripts first."
        )

    return pd.read_csv(path)

# Load the three reconciliation reports
orders = load_report("reconciliation_report.csv")
riders = load_report("rider_cash_reconciliation.csv")
merchants = load_report("merchant_settlement_reconciliation.csv")

exceptions = []

# 1. Order collection exceptions
order_exception_statuses = [
    "MISSING_COLLECTION",
    "AMOUNT_MISMATCH",
    "PAYMENT_PENDING",
    "PAYMENT_FAILED",
    "MATCHED_WITH_DUPLICATE_REFERENCE",
    "COLLECTION_ON_NON_DELIVERED_ORDER",
    "NON_DELIVERED_ORDER_WITH_PAYMENT_ATTEMPT",
]

order_issues = orders[
    orders["reconciliation_status"].isin(order_exception_statuses)
].copy()

for row in order_issues.itertuples(index=False):
    exceptions.append({
        "exception_type": "ORDER_COLLECTION",
        "reference_id": row.order_id,
        "party_id": row.rider_id,
        "date": getattr(row, "order_date", ""),
        "expected_amount": row.expected_cod_amount,
        "actual_amount": row.successful_collected_amount,
        "gap_amount": row.collection_gap,
        "status": row.reconciliation_status,
        "review_required": True,
    })

# 2. Rider cash handover exceptions
rider_issues = riders[
    riders["handover_status"] != "MATCHED"
].copy()

for row in rider_issues.itertuples(index=False):
    exceptions.append({
        "exception_type": "RIDER_CASH_HANDOVER",
        "reference_id": f"{row.rider_id}_{row.business_date}",
        "party_id": row.rider_id,
        "date": row.business_date,
        "expected_amount": row.expected_cash_amount,
        "actual_amount": row.deposited_cash_amount,
        "gap_amount": row.cash_gap,
        "status": row.handover_status,
        "review_required": True,
    })

# 3. Merchant settlement exceptions
merchant_issues = merchants[
    merchants["settlement_status"] != "MATCHED"
].copy()

for row in merchant_issues.itertuples(index=False):
    exceptions.append({
        "exception_type": "MERCHANT_SETTLEMENT",
        "reference_id": f"{row.merchant_id}_{row.period_start}",
        "party_id": row.merchant_id,
        "date": row.period_start,
        "expected_amount": row.expected_payout,
        "actual_amount": row.actual_payout,
        "gap_amount": row.payout_gap,
        "status": row.settlement_status,
        "review_required": True,
    })

columns = [
    "exception_type",
    "reference_id",
    "party_id",
    "date",
    "expected_amount",
    "actual_amount",
    "gap_amount",
    "status",
    "review_required",
]

master = pd.DataFrame(exceptions, columns=columns)

if not master.empty:
    master["expected_amount"] = pd.to_numeric(
        master["expected_amount"], errors="coerce"
    ).round(2)

    master["actual_amount"] = pd.to_numeric(
        master["actual_amount"], errors="coerce"
    ).round(2)

    master["gap_amount"] = pd.to_numeric(
        master["gap_amount"], errors="coerce"
    ).round(2)

    master = master.sort_values(
        ["exception_type", "party_id", "reference_id"]
    )

# Export master exception report
output_file = DATA_DIR / "master_exception_report.csv"
master.to_csv(output_file, index=False)

print("=" * 55)
print("MASTER DELIVERY OPERATIONS REPORT")
print("=" * 55)

print("\nOrder records checked:", len(orders))
print("Rider-date records checked:", len(riders))
print("Merchant-period records checked:", len(merchants))
print("Total exceptions:", len(master))

print("\nExceptions by type:")
if master.empty:
    print("No exceptions found.")
else:
    print(master["exception_type"].value_counts().to_string())

print("\nExceptions by status:")
if master.empty:
    print("No exceptions found.")
else:
    print(master["status"].value_counts().to_string())

print("\nSaved report:", output_file)
print("\nMASTER REPORT CREATED SUCCESSFULLY")
