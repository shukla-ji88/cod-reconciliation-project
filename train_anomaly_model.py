
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = Path("data") / "simulated_cod"
MODEL_DIR = Path("models")

INPUT_FILE = DATA_DIR / "reconciliation_report.csv"
OUTPUT_FILE = DATA_DIR / "ml_anomaly_report.csv"
MODEL_FILE = MODEL_DIR / "cod_isolation_forest.joblib"

# 1. Load reconciliation data
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. Run reconcile_cod.py first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "order_id",
    "order_status",
    "reconciliation_status",
    "expected_cod_amount",
    "successful_collected_amount",
    "collection_gap",
    "collection_record_count",
    "successful_transaction_count",
    "pending_transaction_count",
    "failed_transaction_count",
    "duplicate_reference_count",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 2. Keep delivered orders for model scoring
delivered = df[
    df["order_status"].astype(str).str.upper().eq("DELIVERED")
].copy()

# 3. Define numeric features
# These features describe transaction amounts and collection patterns.
features = [
    "expected_cod_amount",
    "successful_collected_amount",
    "collection_gap",
    "collection_record_count",
    "successful_transaction_count",
    "pending_transaction_count",
    "failed_transaction_count",
    "duplicate_reference_count",
]

for col in features:
    delivered[col] = pd.to_numeric(
        delivered[col], errors="coerce"
    )

if delivered[features].isna().any().any():
    raise ValueError(
        "Missing or invalid numeric feature values found. "
        "Clean the input before training."
    )

# 4. Use matched delivered orders as the normal baseline
baseline = delivered[
    delivered["reconciliation_status"].eq("MATCHED")
].copy()

if len(baseline) < 30:
    raise ValueError(
        f"Only {len(baseline)} matched baseline records found. "
        "At least 30 are required for this demo."
    )

print("Delivered orders to score:", len(delivered))
print("Matched baseline records:", len(baseline))

# 5. Train the unsupervised anomaly-detection model
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)

model.fit(baseline[features])

# 6. Score all delivered orders
# IsolationForest predicts -1 for an anomaly and 1 for a normal pattern.
predictions = model.predict(delivered[features])

# Lower decision scores indicate more unusual observations.
# Negating makes higher anomaly scores mean more unusual.
delivered["anomaly_score"] = (
    -model.score_samples(delivered[features])
)

delivered["ml_prediction"] = [
    "ANOMALY" if prediction == -1 else "NORMAL_PATTERN"
    for prediction in predictions
]

delivered["review_recommended"] = (
    delivered["ml_prediction"].eq("ANOMALY")
)

# 7. Keep non-delivered orders in the report, but do not score them
result = df.copy()
result["anomaly_score"] = float("nan")
result["ml_prediction"] = "NOT_EVALUATED_NON_DELIVERED"
result["review_recommended"] = False

scored = delivered.set_index("order_id")

result = result.set_index("order_id")
result.loc[scored.index, "anomaly_score"] = scored["anomaly_score"]
result.loc[scored.index, "ml_prediction"] = scored["ml_prediction"]
result.loc[scored.index, "review_recommended"] = (
    scored["review_recommended"]
)
result = result.reset_index()

# 8. Save the model and report
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)
result.to_csv(OUTPUT_FILE, index=False)

# 9. Print summary
print("\n" + "=" * 52)
print("COD MACHINE LEARNING ANOMALY REPORT")
print("=" * 52)

print("\nPrediction counts:")
print(delivered["ml_prediction"].value_counts().to_string())

print(
    "\nRule-based exceptions within delivered orders:",
    (~delivered["reconciliation_status"].eq("MATCHED")).sum()
)

print(
    "ML anomalies flagged:",
    delivered["ml_prediction"].eq("ANOMALY").sum()
)

print("\nSaved model:", MODEL_FILE)
print("Saved report:", OUTPUT_FILE)
print("\nANOMALY MODEL TRAINING COMPLETED SUCCESSFULLY")
