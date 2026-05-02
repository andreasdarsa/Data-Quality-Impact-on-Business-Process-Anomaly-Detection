from src.data.loader import load_logs, save_logs
from src.data.schema import validate_logs
from src.data.splitter import split_logs
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

from experiments.configs import (
    SEEDS,
    TEST_SIZE,
    get_raw_log_filename,
    get_train_filename,
    get_test_filename,
)


def make_split_for_seed(seed: int) -> None:
    input_path = RAW_DATA_DIR / get_raw_log_filename(seed)

    logs = load_logs(input_path)
    validate_logs(logs)

    train_logs, test_logs = split_logs(
        logs,
        test_size=TEST_SIZE,
        seed=seed,
        stratify=True,
    )

    save_logs(train_logs, PROCESSED_DATA_DIR / get_train_filename(seed))
    save_logs(test_logs, PROCESSED_DATA_DIR / get_test_filename(seed))

    print(f"[seed={seed}] Loaded: {len(logs)} cases")
    print(f"[seed={seed}] Train: {len(train_logs)} cases")
    print(f"[seed={seed}] Test: {len(test_logs)} cases")


def main() -> None:
    for seed in SEEDS:
        make_split_for_seed(seed)


if __name__ == "__main__":
    main()

