from src.data.adapters.binet_adapter import load_and_convert_binet_log
from src.data.loader import save_logs
from src.data.schema import validate_logs
from src.data.splitter import split_logs
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

from experiments.configs import (
    SEEDS,
    TEST_SIZE,
    BINET_DATA_FILENAME,
    get_binet_train_filename,
    get_binet_test_filename,
)


def make_binet_split_for_seed(seed: int) -> None:
    input_path = RAW_DATA_DIR / BINET_DATA_FILENAME

    logs = load_and_convert_binet_log(input_path)
    validate_logs(logs)

    train_logs, test_logs = split_logs(
        logs,
        test_size=TEST_SIZE,
        seed=seed,
        stratify=True,
    )

    save_logs(
        train_logs,
        PROCESSED_DATA_DIR / get_binet_train_filename(seed),
    )

    save_logs(
        test_logs,
        PROCESSED_DATA_DIR / get_binet_test_filename(seed),
    )

    print(f"[BINet seed={seed}] Loaded: {len(logs)} cases")
    print(f"[BINet seed={seed}] Train: {len(train_logs)} cases")
    print(f"[BINet seed={seed}] Test: {len(test_logs)} cases")

    print(
        f"[BINet seed={seed}] Train anomalies: "
        f"{sum(case['is_anomaly'] for case in train_logs)}"
    )

    print(
        f"[BINet seed={seed}] Test anomalies: "
        f"{sum(case['is_anomaly'] for case in test_logs)}"
    )


def main() -> None:
    for seed in SEEDS:
        make_binet_split_for_seed(seed)


if __name__ == "__main__":
    main()