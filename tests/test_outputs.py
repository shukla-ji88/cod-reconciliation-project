
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "simulated_cod"
MODEL_FILE = BASE_DIR / "models" / "cod_isolation_forest.joblib"


class CODProjectTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the generated reports once for all tests
        cls.orders = pd.read_csv(
            DATA_DIR / "orders.csv"
        )
        cls.collections = pd.read_csv(
            DATA_DIR / "collection_transactions.csv"
        )
        cls.order_report = pd.read_csv(
            DATA_DIR / "reconciliation_report.csv"
        )
        cls.rider_report = pd.read_csv(
            DATA_DIR / "rider_cash_reconciliation.csv"
        )
        cls.merchant_report = pd.read_csv(
            DATA_DIR / "merchant_settlement_reconciliation.csv"
        )
        cls.master_report = pd.read_csv(
            DATA_DIR / "master_exception_report.csv"
        )
        cls.ml_report = pd.read_csv(
            DATA_DIR / "ml_anomaly_report.csv"
        )

    def test_required_project_files_exist(self):
        required_files = [
            BASE_DIR / "data" / "Online Retail.xlsx",
            BASE_DIR / "data" / "cleaned_online_retail.csv",
            DATA_DIR / "orders.csv",
            DATA_DIR / "collection_transactions.csv",
            DATA_DIR / "rider_cash_handovers.csv",
            DATA_DIR / "merchant_settlements.csv",
            DATA_DIR / "reconciliation_report.csv",
            DATA_DIR / "duplicate_transactions.csv",
            DATA_DIR / "unknown_order_collections.csv",
            DATA_DIR / "rider_cash_reconciliation.csv",
            DATA_DIR / "rider_cash_duplicate_transactions.csv",
            DATA_DIR / "merchant_settlement_reconciliation.csv",
            DATA_DIR / "master_exception_report.csv",
            DATA_DIR / "ml_anomaly_report.csv",
            MODEL_FILE,
        ]

        missing = [str(p) for p in required_files if not p.is_file()]
        self.assertEqual(missing, [], f"Missing files: {missing}")

    def test_order_count_and_unique_ids(self):
        self.assertEqual(len(self.orders), 500)
        self.assertTrue(self.orders["order_id"].is_unique)

    def test_order_report_contains_every_order_once(self):
        self.assertEqual(len(self.order_report), len(self.orders))
        self.assertTrue(self.order_report["order_id"].is_unique)
        self.assertSetEqual(
            set(self.order_report["order_id"]),
            set(self.orders["order_id"]),
        )

    def test_order_collection_gap_calculation(self):
        expected_gap = (
            self.order_report["expected_cod_amount"]
            - self.order_report["successful_collected_amount"]
        )

        self.assertTrue(
            np.allclose(
                expected_gap,
                self.order_report["collection_gap"],
                atol=0.011,
            )
        )

    def test_rider_cash_gap_calculation(self):
        expected_gap = (
            self.rider_report["expected_cash_amount"]
            - self.rider_report["deposited_cash_amount"]
        )

        self.assertTrue(
            np.allclose(
                expected_gap,
                self.rider_report["cash_gap"],
                atol=0.011,
            )
        )

    def test_merchant_payout_gap_calculation(self):
        expected_gap = (
            self.merchant_report["expected_payout"]
            - self.merchant_report["actual_payout"]
        )

        self.assertTrue(
            np.allclose(
                expected_gap,
                self.merchant_report["payout_gap"],
                atol=0.011,
            )
        )

    def test_master_report_exception_count(self):
        order_exception_statuses = [
            "MISSING_COLLECTION",
            "AMOUNT_MISMATCH",
            "PAYMENT_PENDING",
            "PAYMENT_FAILED",
            "MATCHED_WITH_DUPLICATE_REFERENCE",
            "COLLECTION_ON_NON_DELIVERED_ORDER",
            "NON_DELIVERED_ORDER_WITH_PAYMENT_ATTEMPT",
        ]

        expected_orders = self.order_report[
            self.order_report["reconciliation_status"].isin(
                order_exception_statuses
            )
        ]

        expected_riders = self.rider_report[
            self.rider_report["handover_status"] != "MATCHED"
        ]

        expected_merchants = self.merchant_report[
            self.merchant_report["settlement_status"] != "MATCHED"
        ]

        expected_count = (
            len(expected_orders)
            + len(expected_riders)
            + len(expected_merchants)
        )

        self.assertEqual(len(self.master_report), expected_count)

    def test_ml_predictions_match_order_status(self):
        self.assertEqual(len(self.ml_report), len(self.orders))
        self.assertTrue(self.ml_report["order_id"].is_unique)

        delivered = self.ml_report[
            self.ml_report["order_status"].eq("DELIVERED")
        ]

        not_delivered = self.ml_report[
            ~self.ml_report["order_status"].eq("DELIVERED")
        ]

        self.assertTrue(
            delivered["ml_prediction"].isin(
                ["ANOMALY", "NORMAL_PATTERN"]
            ).all()
        )

        self.assertTrue(
            not_delivered["ml_prediction"].eq(
                "NOT_EVALUATED_NON_DELIVERED"
            ).all()
        )

        self.assertFalse(
            delivered["anomaly_score"].isna().any()
        )

    def test_saved_ml_model_can_be_loaded(self):
        model = joblib.load(MODEL_FILE)

        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "score_samples"))

    def test_collection_data_has_required_fields(self):
        required_columns = {
            "collection_id",
            "order_id",
            "rider_id",
            "payment_mode",
            "payment_status",
            "collected_amount",
            "transaction_reference",
        }

        self.assertTrue(
            required_columns.issubset(
                set(self.collections.columns)
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
