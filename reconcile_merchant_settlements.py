
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data") / "simulated_cod"

orders_file = DATA_DIR / "orders.csv"
settlements_file = DATA_DIR / "merchant_settlements.csv"

# 1. Check required files
for file in [orders_file, settlements_file]:
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

# 2. Load data
orders = pd.read_csv(orders_file)
settlements = pd.read_csv(settlements_file)

# 3. Validate required columns
required_orders = {
    "order_id", "order_date", "order_status",
    "merchant_id", "cod_amount", "commission_rate"
}

required_settlements = {
    "settlement_id", "merchant_id", "period_start",
    "payout_amount", "settlement_reference"
}

if not required_orders.issubset(orders.columns):
    raise ValueError("Required columns missing from orders.csv")

if not required_settlements.issubset(settlements.columns):
    raise ValueError(
        "Required columns missing from merchant_settlements.csv"
    )

if orders["order_id"].duplicated().any():
    raise ValueError("Duplicate order IDs found")

if settlements["settlement_id"].duplicated().any():
    raise ValueError("Duplicate settlement IDs found")

# 4. Convert data types
orders["order_date"] = pd.to_datetime(
    orders["order_date"], errors="coerce"
)
orders["cod_amount"] = pd.to_numeric(
    orders["cod_amount"], errors="coerce"
)
orders["commission_rate"] = pd.to_numeric(
    orders["commission_rate"], errors="coerce"
)

settlements["period_start"] = pd.to_datetime(
    settlements["period_start"], errors="coerce"
)
settlements["payout_amount"] = pd.to_numeric(
    settlements["payout_amount"], errors="coerce"
)

if orders[
    ["order_date", "cod_amount", "commission_rate"]
].isna().any().any():
    raise ValueError("Invalid order dates, amounts or commission rates")

if settlements[
    ["period_start", "payout_amount"]
].isna().any().any():
    raise ValueError("Invalid settlement dates or payout amounts")

if not orders["commission_rate"].between(0, 1).all():
    raise ValueError("Commission rates must be between 0 and 1")

if (orders["cod_amount"] < 0).any():
    raise ValueError("Negative COD amount found in orders")

if (settlements["payout_amount"] < 0).any():
    raise ValueError("Negative payout found; review the source data")

# 5. Keep delivered orders for payout calculation
delivered = orders.loc[
    orders["order_status"].astype(str).str.upper().eq("DELIVERED")
].copy()

# 6. Create weekly settlement periods.
# The period starts on Monday.
delivered["period_start"] = (
    delivered["order_date"]
    - pd.to_timedelta(
        delivered["order_date"].dt.weekday, unit="D"
    )
).dt.normalize()

# 7. Calculate expected payout for every eligible order
delivered["expected_payout"] = (
    delivered["cod_amount"]
    * (1 - delivered["commission_rate"])
)

expected = (
    delivered.groupby(
        ["merchant_id", "period_start"], as_index=False
    )
    .agg(
        delivered_order_count=("order_id", "nunique"),
        gross_delivered_cod=("cod_amount", "sum"),
        expected_commission=(
            "cod_amount",
            lambda values: 0.0
        )
    )
)

# Calculate commission explicitly for accurate reporting
commission_by_period = (
    delivered.assign(
        commission_amount=(
            delivered["cod_amount"]
            * delivered["commission_rate"]
        )
    )
    .groupby(["merchant_id", "period_start"], as_index=False)
    ["commission_amount"].sum()
)

expected = expected.drop(columns=["expected_commission"]).merge(
    commission_by_period,
    on=["merchant_id", "period_start"],
    how="left",
    validate="one_to_one"
)

expected = expected.rename(
    columns={"commission_amount": "expected_commission"}
)

# Sum expected merchant payout
payout_expected = (
    delivered.groupby(
        ["merchant_id", "period_start"], as_index=False
    )["expected_payout"].sum()
)

expected = expected.merge(
    payout_expected,
    on=["merchant_id", "period_start"],
    how="left",
    validate="one_to_one"
)

# 8. Aggregate recorded payouts by merchant and period
actual = (
    settlements.groupby(
        ["merchant_id", "period_start"], as_index=False
    )
    .agg(
        settlement_record_count=("settlement_id", "count"),
        actual_payout=("payout_amount", "sum")
    )
)

# 9. Match expected and actual amounts.
# Outer join preserves both missing and unexpected settlements.
report = expected.merge(
    actual,
    on=["merchant_id", "period_start"],
    how="outer",
    validate="one_to_one",
    indicator=True
)

# 10. Fill missing amounts and counts
for column in [
    "delivered_order_count",
    "gross_delivered_cod",
    "expected_commission",
    "expected_payout",
    "settlement_record_count",
    "actual_payout"
]:
    if column not in report.columns:
        report[column] = 0
    report[column] = report[column].fillna(0)

report["delivered_order_count"] = (
    report["delivered_order_count"].astype(int)
)
report["settlement_record_count"] = (
    report["settlement_record_count"].astype(int)
)

# Expected less actual: positive = short payout
report["payout_gap"] = (
    report["expected_payout"] - report["actual_payout"]
).round(2)

# 11. Classify settlement status
def classify_settlement(row):
    expected_amount = row["expected_payout"]
    actual_amount = row["actual_payout"]

    if expected_amount > 0 and row["settlement_record_count"] == 0:
        return "MISSING_SETTLEMENT"

    if expected_amount == 0 and actual_amount > 0:
        return "UNEXPECTED_SETTLEMENT"

    if abs(row["payout_gap"]) <= 0.01:
        return "MATCHED"

    return "PAYOUT_MISMATCH"

report["settlement_status"] = report.apply(
    classify_settlement, axis=1
)

report["period_start"] = pd.to_datetime(
    report["period_start"]
).dt.strftime("%Y-%m-%d")

report_path = DATA_DIR / "merchant_settlement_reconciliation.csv"
report.to_csv(report_path, index=False)

# 12. Print summary
print("=" * 55)
print("MERCHANT SETTLEMENT RECONCILIATION")
print("=" * 55)

print("\nMerchant-period records:", len(report))
print("\nSettlement status:")
print(report["settlement_status"].value_counts().to_string())

print(
    "\nGross delivered COD: ₹",
    f"{report['gross_delivered_cod'].sum():,.2f}"
)
print(
    "Expected merchant commission: ₹",
    f"{report['expected_commission'].sum():,.2f}"
)
print(
    "Expected merchant payout: ₹",
    f"{report['expected_payout'].sum():,.2f}"
)
print(
    "Actual recorded merchant payout: ₹",
    f"{report['actual_payout'].sum():,.2f}"
)
print(
    "Net payout gap: ₹",
    f"{report['payout_gap'].sum():,.2f}"
)

print("\nSaved report:", report_path)
print("\nMERCHANT RECONCILIATION COMPLETED SUCCESSFULLY")
