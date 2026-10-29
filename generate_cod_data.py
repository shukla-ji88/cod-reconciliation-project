
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible simulated delivery operations data
rng = np.random.default_rng(42)

OUTPUT_DIR = Path("data") / "simulated_cod"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Generate orders
# --------------------------------------------------
n_orders = 500

merchants = [f"M{i:03d}" for i in range(1, 21)]
riders = [f"R{i:03d}" for i in range(1, 13)]

order_dates = (
    pd.Timestamp("2026-09-01")
    + pd.to_timedelta(rng.integers(0, 30, n_orders), unit="D")
)

merchant_commission = {
    merchant: round(float(rate), 3)
    for merchant, rate in zip(
        merchants, rng.uniform(0.05, 0.12, len(merchants))
    )
}

orders = pd.DataFrame({
    "order_id": [f"O{i:05d}" for i in range(1, n_orders + 1)],
    "order_date": order_dates,
    "merchant_id": rng.choice(merchants, n_orders),
    "rider_id": rng.choice(riders, n_orders),
    "order_status": rng.choice(
        ["DELIVERED", "CANCELLED", "RETURNED"],
        n_orders,
        p=[0.88, 0.07, 0.05]
    ),
    "cod_amount": rng.integers(100, 3001, n_orders).astype(float)
})

orders["commission_rate"] = orders["merchant_id"].map(
    merchant_commission
)

# --------------------------------------------------
# 2. Generate collection transactions
# Only delivered orders are eligible in this demo.
# Some records are deliberately missing or mismatched.
# --------------------------------------------------
base_collections = []
transaction_number = 1

delivered_orders = orders[
    orders["order_status"] == "DELIVERED"
]

for order in delivered_orders.itertuples(index=False):

    # Simulate a missing collection record
    if rng.random() < 0.03:
        continue

    amount = float(order.cod_amount)

    # Simulate under/over collection
    if rng.random() < 0.05:
        difference = int(rng.integers(10, 101))
        amount += float(rng.choice([-1, 1]) * difference)

    payment_mode = str(
        rng.choice(["CASH", "QR"], p=[0.55, 0.45])
    )

    payment_status = str(
        rng.choice(
            ["SUCCESS", "FAILED", "PENDING"],
            p=[0.96, 0.025, 0.015]
        )
    )

    base_collections.append({
        "collection_id": f"C{transaction_number:05d}",
        "order_id": order.order_id,
        "rider_id": order.rider_id,
        "payment_mode": payment_mode,
        "collected_amount": round(max(0, amount), 2),
        "payment_status": payment_status,
        "transaction_reference": f"TXN{transaction_number:06d}",
        "collection_time": (
            order.order_date
            + pd.Timedelta(hours=int(rng.integers(9, 23)))
        )
    })

    transaction_number += 1

base_df = pd.DataFrame(base_collections)

# Deliberately duplicate some transaction references.
# This helps test duplicate detection later.
duplicate_count = max(1, round(len(base_df) * 0.015))

if not base_df.empty:
    duplicate_indices = rng.choice(
        base_df.index,
        size=min(duplicate_count, len(base_df)),
        replace=False
    )

    duplicate_rows = base_df.loc[duplicate_indices].copy()

    duplicate_rows["collection_id"] = [
        f"C{transaction_number + i:05d}"
        for i in range(len(duplicate_rows))
    ]

    transaction_number += len(duplicate_rows)

    collections = pd.concat(
        [base_df, duplicate_rows],
        ignore_index=True
    )
else:
    collections = base_df.copy()

# --------------------------------------------------
# 3. Generate rider cash handovers
# Based on original successful CASH records,
# not the artificially duplicated rows.
# --------------------------------------------------
cash = base_df[
    (base_df["payment_mode"] == "CASH")
    & (base_df["payment_status"] == "SUCCESS")
].copy()

handovers = []

if not cash.empty:
    cash["handover_date"] = cash["collection_time"].dt.normalize()

    daily_cash = (
        cash.groupby(["rider_id", "handover_date"], as_index=False)
        ["collected_amount"].sum()
    )

    handover_number = 1

    for row in daily_cash.itertuples(index=False):

        # Simulate a missing cash handover record
        if rng.random() < 0.02:
            continue

        deposited_amount = float(row.collected_amount)

        # Simulate a deposit discrepancy
        if rng.random() < 0.08:
            difference = float(rng.integers(20, 201))
            deposited_amount += float(
                rng.choice([-1, 1]) * difference
            )

        handovers.append({
            "handover_id": f"H{handover_number:05d}",
            "rider_id": row.rider_id,
            "handover_date": row.handover_date,
            "deposited_cash_amount": round(
                max(0, deposited_amount), 2
            )
        })

        handover_number += 1

handover_df = pd.DataFrame(
    handovers,
    columns=[
        "handover_id",
        "rider_id",
        "handover_date",
        "deposited_cash_amount"
    ]
)

# --------------------------------------------------
# 4. Generate merchant settlements
# Demo assumption: delivered COD amount less the
# merchant's configured commission is payable.
# Actual contracts can have other fee rules.
# --------------------------------------------------
settlement_orders = delivered_orders.copy()

settlement_orders["period_start"] = (
    settlement_orders["order_date"]
    - pd.to_timedelta(
        settlement_orders["order_date"].dt.weekday,
        unit="D"
    )
)

settlement_orders["expected_payout"] = (
    settlement_orders["cod_amount"]
    * (1 - settlement_orders["commission_rate"])
)

weekly = (
    settlement_orders
    .groupby(["merchant_id", "period_start"], as_index=False)
    ["expected_payout"].sum()
)

settlements = []
settlement_number = 1

for row in weekly.itertuples(index=False):

    # Simulate a missing payout record
    if rng.random() < 0.03:
        continue

    payout = float(row.expected_payout)

    # Simulate a short or excess payout
    if rng.random() < 0.08:
        payout *= float(rng.choice([0.95, 1.05]))

    settlements.append({
        "settlement_id": f"S{settlement_number:05d}",
        "merchant_id": row.merchant_id,
        "period_start": row.period_start,
        "settlement_date": row.period_start + pd.Timedelta(days=7),
        "payout_amount": round(payout, 2),
        "settlement_reference": f"UTR{settlement_number:07d}"
    })

    settlement_number += 1

settlement_df = pd.DataFrame(
    settlements,
    columns=[
        "settlement_id",
        "merchant_id",
        "period_start",
        "settlement_date",
        "payout_amount",
        "settlement_reference"
    ]
)

# --------------------------------------------------
# 5. Save all datasets
# --------------------------------------------------
files = {
    "orders.csv": orders,
    "collection_transactions.csv": collections,
    "rider_cash_handovers.csv": handover_df,
    "merchant_settlements.csv": settlement_df
}

print("DATA SOURCE: SIMULATED DELIVERY OPERATIONS DATA")
print("Seed: 42")

for filename, dataframe in files.items():
    path = OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"{filename}: {len(dataframe)} rows | Saved: {path}")

print("\nOrders by status:")
print(orders["order_status"].value_counts())

print("\nCollection payment modes:")
if collections.empty:
    print("No collection records generated.")
else:
    print(collections["payment_mode"].value_counts())

print("\nAll simulated datasets generated successfully!")
