
import unittest
import numpy as np

from src.data.adapters.bpic13_adapter import (
    load_and_convert_bpic13_log,
)
from src.data.schema import validate_logs
from src.data.splitter import split_logs
from src.process_mining import build_dfg, prune_dfg
from src.features import extract_features


DATASET_PATH = "data/raw/bpic13-0.3-1.json"

SPLIT_SEED = 42
TEST_SIZE = 0.3
MIN_DFG_FREQUENCY = 2


class TestBPIC13Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Load and validate dataset
        cls.logs = load_and_convert_bpic13_log(DATASET_PATH)
        validate_logs(cls.logs)

        # 2. Split into training and test sets
        cls.train_logs, cls.test_logs = split_logs(
            cls.logs,
            test_size=TEST_SIZE,
            seed=SPLIT_SEED,
            stratify=True,
        )

        # 3. Keep only normal training cases for DFG
        cls.normal_train_logs = [
            case for case in cls.train_logs
            if not case["is_anomaly"]
        ]

        # 4. Build and prune DFG
        cls.dfg = build_dfg(cls.normal_train_logs)
        cls.dfg = prune_dfg(
            cls.dfg,
            min_freq=MIN_DFG_FREQUENCY,
        )

        # 5. Extract features
        cls.X_train, cls.y_train, cls.train_subtypes = (
            extract_features(cls.train_logs, cls.dfg)
        )

        cls.X_test, cls.y_test, cls.test_subtypes = (
            extract_features(cls.test_logs, cls.dfg)
        )

    def test_dataset_counts(self):
        self.assertEqual(len(self.logs), 1487)

        self.assertEqual(
            sum(len(case["events"]) for case in self.logs),
            6891,
        )

        self.assertEqual(
            sum(case["is_anomaly"] for case in self.logs),
            362,
        )

    def test_split_integrity(self):
        self.assertEqual(
            len(self.train_logs) + len(self.test_logs),
            len(self.logs),
        )

        train_ids = {
            case["case_id"] for case in self.train_logs
        }
        test_ids = {
            case["case_id"] for case in self.test_logs
        }

        self.assertTrue(train_ids.isdisjoint(test_ids))

    def test_stratification(self):
        train_anomalies = sum(
            case["is_anomaly"] for case in self.train_logs
        )

        test_anomalies = sum(
            case["is_anomaly"] for case in self.test_logs
        )

        self.assertGreater(train_anomalies, 0)
        self.assertGreater(test_anomalies, 0)

        self.assertLess(
            abs(
                train_anomalies / len(self.train_logs)
                - test_anomalies / len(self.test_logs)
            ),
            0.02,
        )

    def test_dfg_uses_only_normal_training_cases(self):
        self.assertTrue(len(self.normal_train_logs) > 0)

        expected_dfg = prune_dfg(
            build_dfg(self.normal_train_logs),
            min_freq=MIN_DFG_FREQUENCY,
        )

        self.assertEqual(self.dfg, expected_dfg)

    def test_feature_columns_match(self):
        self.assertEqual(
            list(self.X_train.columns),
            list(self.X_test.columns),
        )

    def test_feature_matrices_are_numeric(self):
        self.assertTrue(
            all(
                np.issubdtype(dtype, np.number)
                for dtype in self.X_train.dtypes
            )
        )

        self.assertTrue(
            all(
                np.issubdtype(dtype, np.number)
                for dtype in self.X_test.dtypes
            )
        )

    def test_no_nan_or_infinite_values(self):
        self.assertTrue(
            np.isfinite(
                self.X_train.to_numpy(dtype=float)
            ).all()
        )

        self.assertTrue(
            np.isfinite(
                self.X_test.to_numpy(dtype=float)
            ).all()
        )

    def test_labels_match_cases(self):
        expected_train_labels = [
            int(case["is_anomaly"])
            for case in self.train_logs
        ]

        expected_test_labels = [
            int(case["is_anomaly"])
            for case in self.test_logs
        ]

        self.assertEqual(
            self.y_train.tolist(),
            expected_train_labels,
        )

        self.assertEqual(
            self.y_test.tolist(),
            expected_test_labels,
        )

    def test_feature_matrix_shapes(self):
        self.assertEqual(
            self.X_train.shape[0],
            len(self.train_logs),
        )

        self.assertEqual(
            self.X_test.shape[0],
            len(self.test_logs),
        )

        self.assertGreater(self.X_train.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
