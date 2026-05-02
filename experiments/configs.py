from src.degradation import DegradationConfig


SEEDS = [42, 43, 44, 45, 46]

RAW_LOG_FILENAME_TEMPLATE = "baseline_event_log_seed_{seed}.json"

TRAIN_FILENAME_TEMPLATE = "train_seed_{seed}.json"
TEST_FILENAME_TEMPLATE = "test_seed_{seed}.json"

TEST_SIZE = 0.3

THRESHOLD_PERCENTILE = 90


def get_raw_log_filename(seed: int) -> str:
    return RAW_LOG_FILENAME_TEMPLATE.format(seed=seed)


def get_train_filename(seed: int) -> str:
    return TRAIN_FILENAME_TEMPLATE.format(seed=seed)


def get_test_filename(seed: int) -> str:
    return TEST_FILENAME_TEMPLATE.format(seed=seed)


def get_degradation_configs(seed: int) -> list[DegradationConfig]:
    return [
        # Completeness
        DegradationConfig("completeness", 0.1, variant="random", seed=seed),
        DegradationConfig("completeness", 0.2, variant="random", seed=seed),
        DegradationConfig("completeness", 0.3, variant="random", seed=seed),

        # Accuracy
        DegradationConfig("accuracy", 0.1, variant="duration_timestamp", seed=seed),
        DegradationConfig("accuracy", 0.2, variant="duration_timestamp", seed=seed),
        DegradationConfig("accuracy", 0.3, variant="duration_timestamp", seed=seed),

        # Consistency
        DegradationConfig("consistency", 0.1, variant="resource_mismatch", seed=seed),
        DegradationConfig("consistency", 0.2, variant="resource_mismatch", seed=seed),
        DegradationConfig("consistency", 0.3, variant="resource_mismatch", seed=seed),

        # Timeliness
        DegradationConfig("timeliness", 0.1, variant="event_delay", seed=seed),
        DegradationConfig("timeliness", 0.2, variant="event_delay", seed=seed),
        DegradationConfig("timeliness", 0.3, variant="event_delay", seed=seed),
    ]