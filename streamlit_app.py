
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "simulated_cod"

st.set_page_config(
    page_title="COD Operations Intelligence",
    page_icon="📦",
    layout="wide",
)

st.title("📦 COD Operations Intelligence Platform")
st.caption(
    "Hyperlocal delivery reconciliation | "
    "Simulated delivery operations data"
)

FILES = {
    "orders": "reconciliation_report.csv",
    "riders": "rider_cash_reconciliation.csv",
    "merchants": "merchant_settlement_reconciliation.csv",
    "master": "master_exception_report.csv",
    "ml": "ml_anomaly_report.csv",
}


@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string)


# --------------------------------------------------
# Load reports
# --------------------------------------------------
missing_files = [
    filename
    for filename in FILES.values()
    if not (DATA_DIR / filename).exists()
]

if missing_files:
    st.error("Required report files are missing:")
    for filename in missing_files:
        st.write(f"- {filename}")

    st.info(
        "Run the reconciliation scripts and "
        "train_anomaly_model.py before opening the dashboard."
    )
    st.stop()

orders = load_csv(str(DATA_DIR / FILES["orders"]))
riders = load_csv(str(DATA_DIR / FILES["riders"]))
merchants = load_csv(str(DATA_DIR / FILES["merchants"]))
master = load_csv(str(DATA_DIR / FILES["master"]))
ml = load_csv(str(DATA_DIR / FILES["ml"]))

# --------------------------------------------------
# Type normalization
# --------------------------------------------------
for df, col in [
    (orders, "order_date"),
    (riders, "business_date"),
    (merchants, "period_start"),
]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

numeric_columns = {
    "orders": [
        "expected_cod_amount",
        "successful_collected_amount",
        "collection_gap",
    ],
    "riders": [
        "expected_cash_amount",
        "deposited_cash_amount",
        "cash_gap",
    ],
    "merchants": [
        "gross_delivered_cod",
        "expected_commission",
        "expected_payout",
        "actual_payout",
        "payout_gap",
    ],
}

for df, columns in [
    (orders, numeric_columns["orders"]),
    (riders, numeric_columns["riders"]),
    (merchants, numeric_columns["merchants"]),
]:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

def options_for(df, column):
    if column not in df.columns:
        return ["All"]

    values = sorted(
        df[column].dropna().astype(str).unique().tolist()
    )
    return ["All"] + values


merchant_choice = st.sidebar.selectbox(
    "Merchant",
    options_for(orders, "merchant_id"),
)

rider_choice = st.sidebar.selectbox(
    "Rider",
    options_for(orders, "rider_id"),
)

status_choice = st.sidebar.selectbox(
    "Order status",
    options_for(orders, "order_status"),
)

filtered_orders = orders.copy()

if merchant_choice != "All":
    filtered_orders = filtered_orders[
        filtered_orders["merchant_id"].astype(str) == merchant_choice
    ]

if rider_choice != "All":
    filtered_orders = filtered_orders[
        filtered_orders["rider_id"].astype(str) == rider_choice
    ]

if status_choice != "All":
    filtered_orders = filtered_orders[
        filtered_orders["order_status"].astype(str) == status_choice
    ]

selected_ids = set(filtered_orders["order_id"].astype(str))

filtered_ml = ml[
    ml["order_id"].astype(str).isin(selected_ids)
].copy()

filtered_riders = riders.copy()
if rider_choice != "All":
    filtered_riders = filtered_riders[
        filtered_riders["rider_id"].astype(str) == rider_choice
    ]

filtered_merchants = merchants.copy()
if merchant_choice != "All":
    filtered_merchants = filtered_merchants[
        filtered_merchants["merchant_id"].astype(str)
        == merchant_choice
    ]

# --------------------------------------------------
# 1. Overview KPIs
# --------------------------------------------------
st.header("1. COD Collection Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Orders in scope",
    f"{len(filtered_orders):,}",
)

expected_cod = filtered_orders["expected_cod_amount"].sum()
col2.metric(
    "Expected COD",
    f"₹{expected_cod:,.2f}",
)

collected = filtered_orders["successful_collected_amount"].sum()
col3.metric(
    "Recorded successful collections",
    f"₹{collected:,.2f}",
)

collection_gap = filtered_orders["collection_gap"].sum()
col4.metric(
    "Net collection gap",
    f"₹{collection_gap:,.2f}",
)

st.caption(
    "Positive collection gap = expected amount exceeds recorded "
    "successful collections. Negative gap = recorded collections exceed "
    "the expected amount. These are reconciliation differences, not "
    "confirmed losses."
)

# --------------------------------------------------
# 2. Order reconciliation
# --------------------------------------------------
st.header("2. Order Reconciliation")

if "reconciliation_status" in filtered_orders.columns:
    status_counts = (
        filtered_orders["reconciliation_status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Orders")
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Orders by reconciliation status")
        st.bar_chart(status_counts.set_index("Status"))

    with right:
        st.subheader("Order exception details")
        exception_orders = filtered_orders[
            filtered_orders["reconciliation_status"] != "MATCHED"
        ]
        st.dataframe(
            exception_orders,
            use_container_width=True,
            hide_index=True,
        )

st.download_button(
    "Download filtered order report",
    data=filtered_orders.to_csv(index=False).encode("utf-8"),
    file_name="filtered_cod_reconciliation.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 3. Rider cash reconciliation
# --------------------------------------------------
st.header("3. Rider Cash Handover")

r1, r2, r3 = st.columns(3)

r1.metric(
    "Expected cash",
    f"₹{filtered_riders['expected_cash_amount'].sum():,.2f}",
)

r2.metric(
    "Recorded cash deposits",
    f"₹{filtered_riders['deposited_cash_amount'].sum():,.2f}",
)

r3.metric(
    "Net cash gap",
    f"₹{filtered_riders['cash_gap'].sum():,.2f}",
)

if not filtered_riders.empty:
    st.bar_chart(
        filtered_riders
        .groupby("rider_id")["cash_gap"]
        .sum()
        .sort_values(ascending=False)
    )

    st.dataframe(
        filtered_riders[
            filtered_riders["handover_status"] != "MATCHED"
        ],
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download rider cash report",
    data=filtered_riders.to_csv(index=False).encode("utf-8"),
    file_name="rider_cash_reconciliation.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 4. Merchant settlements
# --------------------------------------------------
st.header("4. Merchant Settlement Reconciliation")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Expected payout",
    f"₹{filtered_merchants['expected_payout'].sum():,.2f}",
)

m2.metric(
    "Actual recorded payout",
    f"₹{filtered_merchants['actual_payout'].sum():,.2f}",
)

m3.metric(
    "Net payout gap",
    f"₹{filtered_merchants['payout_gap'].sum():,.2f}",
)

if not filtered_merchants.empty:
    st.dataframe(
        filtered_merchants[
            filtered_merchants["settlement_status"] != "MATCHED"
        ],
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download merchant settlement report",
    data=filtered_merchants.to_csv(index=False).encode("utf-8"),
    file_name="merchant_settlement_reconciliation.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 5. Machine learning review
# --------------------------------------------------
st.header("5. Machine Learning Anomaly Review")

if "ml_prediction" in filtered_ml.columns:
    anomaly_count = (
        filtered_ml["ml_prediction"] == "ANOMALY"
    ).sum()

    normal_count = (
        filtered_ml["ml_prediction"] == "NORMAL_PATTERN"
    ).sum()

    non_delivered_count = (
        filtered_ml["ml_prediction"] == "NOT_EVALUATED_NON_DELIVERED"
    ).sum()

    a1, a2, a3 = st.columns(3)

    a1.metric("ML anomalies flagged", f"{anomaly_count:,}")
    a2.metric("Normal-pattern records", f"{normal_count:,}")
    a3.metric("Not evaluated", f"{non_delivered_count:,}")

    anomaly_details = filtered_ml[
        filtered_ml["ml_prediction"] == "ANOMALY"
    ].copy()

    st.subheader("Flagged orders for manual review")
    st.dataframe(
        anomaly_details.sort_values(
            "anomaly_score", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download ML anomaly report",
        data=filtered_ml.to_csv(index=False).encode("utf-8"),
        file_name="ml_anomaly_review.csv",
        mime="text/csv",
    )

st.caption(
    "The datasets used here are simulated delivery operations data. "
    "An ML anomaly is a review signal, not proof of fraud. "
    "Collection gaps, cash gaps and settlement gaps describe different "
    "stages and must not simply be added to calculate business loss."
)

st.success("COD Operations Intelligence dashboard loaded successfully.")
