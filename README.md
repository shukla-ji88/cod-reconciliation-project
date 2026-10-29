
# COD Operations Intelligence Platform

An end-to-end Data Science project for hyperlocal delivery operations, focused on Cash on Delivery (COD) reconciliation, rider cash handovers, merchant settlements, and transaction anomaly detection.

> **Data disclaimer:** The delivery-operation ledgers in this project are simulated for educational and testing purposes. They do not represent actual riders, merchants, payments, or financial losses.

## Business Problem

Delivery businesses need to reconcile the amount expected from COD orders with the collections recorded, cash deposited by riders, and payouts recorded for merchants.

This project helps operations teams identify discrepancies and prioritize records for manual review.

## Key Features

- Public retail transaction dataset analysis and cleaning
- Order-level COD collection reconciliation
- Duplicate payment-reference detection
- Rider-wise cash handover reconciliation
- Merchant settlement reconciliation
- Master exception report across reconciliation modules
- Isolation Forest anomaly detection
- Interactive Streamlit dashboard
- CSV exports for operational review
- Automated tests for key calculations and output files

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Excel/CSV data processing
- Git and GitHub

## Data Sources

### Public dataset

The project uses the UCI Online Retail dataset for real-world retail transaction analysis.

Official source:
https://archive.ics.uci.edu/dataset/352/online+retail

The public retail dataset does not contain rider COD collections, cash handovers, QR payment records, or merchant settlement ledgers.

### Simulated delivery operations data

The project generates separate simulated datasets for:

- Orders
- Collection transactions
- Rider cash handovers
- Merchant settlements

The simulated ledgers include deliberately introduced test cases to exercise reconciliation rules. They must not be interpreted as real-world incident rates or fraud evidence.

## Project Structure

```text
COD-Reconciliation-project/
├── data/
│   ├── Online Retail.xlsx              # Download separately; not committed
│   ├── cleaned_online_retail.csv       # Generated locally; not committed
│   └── simulated_cod/
│       ├── orders.csv
│       ├── collection_transactions.csv
│       ├── rider_cash_handovers.csv
│       ├── merchant_settlements.csv
│       ├── reconciliation_report.csv
│       ├── duplicate_transactions.csv
│       ├── unknown_order_collections.csv
│       ├── rider_cash_reconciliation.csv
│       ├── rider_cash_duplicate_transactions.csv
│       ├── merchant_settlement_reconciliation.csv
│       ├── master_exception_report.csv
│       └── ml_anomaly_report.csv
├── models/
│   └── cod_isolation_forest.joblib
├── tests/
│   └── test_outputs.py
├── check_dataset.py
├── data_quality_check.py
├── clean_retail_data.py
├── generate_cod_data.py
├── reconcile_cod.py
├── reconcile_rider_cash.py
├── reconcile_merchant_settlements.py
├── build_master_report.py
├── train_anomaly_model.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Some generated files may be absent until the corresponding script is run.


## Setup

### 1. Install Python

Use a Python version compatible with the installed scientific Python packages.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the public dataset

Download the UCI Online Retail dataset from the official source above.

Place the extracted Excel file here:

```text
data/Online Retail.xlsx
```

Create the `data` folder first if it does not exist.

## Run the Project

Run these commands from the project root in the following order:

```powershell
python check_dataset.py
python data_quality_check.py
python clean_retail_data.py
python generate_cod_data.py
python reconcile_cod.py
python reconcile_rider_cash.py
python reconcile_merchant_settlements.py
python build_master_report.py
python train_anomaly_model.py
```

### Run automated tests

```powershell
python -m unittest discover -s tests -v
```

The test suite checks required output files, order uniqueness, gap calculations, master report counts, and saved ML model loading.

Run the tests after generating the datasets and reports.

### Launch the dashboard

```powershell
python -m streamlit run streamlit_app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

## Reconciliation Logic

### Order-level COD

Collection gap:

`Expected COD - Successful recorded collections`

Positive values indicate that the expected amount exceeds the successfully recorded collection amount. Negative values indicate the reverse.

### Rider cash handover

Cash gap:

`Expected successful cash - Recorded cash deposits`

The initial demo compares cash collections with deposits grouped by rider and date.

### Merchant settlement

Expected merchant payout:

`Delivered COD - Configured merchant commission`

Actual contractual settlement rules may also involve refunds, taxes, fees, adjustments, payout schedules, and other agreed deductions. Those are outside this initial demo's simplified calculation.

### Machine Learning

The project uses Isolation Forest to identify unusual patterns among delivered orders.

An anomaly score is a review signal, not proof of fraud. The model requires further evaluation on known test cases and held-out data before any operational use.

## Dashboard

The Streamlit dashboard displays:

- Order-level collection KPIs
- Collection exception details
- Rider cash handover differences
- Merchant payout mismatches
- ML anomaly review results
- Filtered CSV downloads

## Limitations

- Delivery ledgers are simulated, not real payment-provider records.
- Public retail transactions do not validate actual COD reconciliation.
- Cash handover matching currently uses rider and date rather than bank-verified deposit references.
- The merchant payout formula represents a simplified demo assumption.
- The ML model has not been validated for production decisions.
- Financial discrepancies must be investigated against original transaction and settlement evidence.

## Future Improvements

- Import bank and QR payment-provider statements
- Match settlements using transaction references and settlement batch IDs
- Add a database backend
- Add role-based access controls and audit logs
- Improve exception prioritization and model evaluation
- Deploy the dashboard with appropriate authentication
- Add configurable fees, refunds, and settlement schedules

## License and Data Attribution

Check the terms of the source dataset before redistributing it. This repository's generated delivery ledgers are simulated and are not official payment-provider or merchant records.
