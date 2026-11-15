
from src.data.adapters.bpic13_adapter import load_and_convert_bpic13_log
from src.data import split_logs, save_logs
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

from experiments.configs import (
    BPIC13_DATA_FILENAME,
    SPLIT_SEEDS,
    TEST_SIZE,
    get_bpic13_train_filename,
    get_bpic13_test_filename,
)


def run_bpic13_splits() -> None:
    input_path = RAW_DATA_DIR / BPIC13_DATA_FILENAME

    print(f"Loading BPIC13 dataset: {input_path}")

    logs = load_and_convert_bpic13_log(input_path)

    print(f"Total cases: {len(logs)}")
    print(
        f"Normal cases: "
        f"{sum(not case['is_anomaly'] for case in logs)}"
    )
    print(
        f"Anomalous cases: "
        f"{sum(case['is_anomaly'] for case in logs)}"
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SPLIT_SEEDS:
        train_logs, test_logs = split_logs(
            logs=logs,
            test_size=TEST_SIZE,
            seed=seed,
            stratify=True,
        )

        train_path = (
            PROCESSED_DATA_DIR /
            get_bpic13_train_filename(seed)
        )

        test_path = (
            PROCESSED_DATA_DIR /
            get_bpic13_test_filename(seed)
        )

        save_logs(train_logs, train_path)
        save_logs(test_logs, test_path)

        train_anomalies = sum(
            case["is_anomaly"] for case in train_logs
        )
        test_anomalies = sum(
            case["is_anomaly"] for case in test_logs
        )

        print(f"\n[BPIC13 seed={seed}]")
        print(f"Train cases: {len(train_logs)}")
        print(f"Test cases: {len(test_logs)}")
        print(f"Train anomalies: {train_anomalies}")
        print(f"Test anomalies: {test_anomalies}")
        print(f"Saved train: {train_path}")
        print(f"Saved test: {test_path}")

    print("\nAll BPIC13 splits generated successfully.")


if __name__ == "__main__":
    run_bpic13_splits()
