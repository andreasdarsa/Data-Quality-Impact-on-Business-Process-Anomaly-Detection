from src.degradation import DegradationConfig


# Train/test split seeds
SPLIT_SEEDS = [42, 43, 44, 45, 46]

# Independent degradation repetitions
DEGRADATION_SEEDS = [101, 102, 103, 104, 105]

N_REPETITIONS = 5

BINET_DATA_FILENAME = "bpic12-0.3-1.json"

BINET_TRAIN_FILENAME_TEMPLATE = "binet_train_seed_{seed}.json"
BINET_TEST_FILENAME_TEMPLATE = "binet_test_seed_{seed}.json"

TEST_SIZE = 0.3

THRESHOLD_PERCENTILE = 90

DEGRADATION_LEVELS = [0.1, 0.2, 0.3]


def get_binet_train_filename(seed: int) -> str:
    return BINET_TRAIN_FILENAME_TEMPLATE.format(seed=seed)


def get_binet_test_filename(seed: int) -> str:
    return BINET_TEST_FILENAME_TEMPLATE.format(seed=seed)


def get_degradation_configs(
    degradation_seed: int
) -> list[DegradationConfig]:

    configs = []

    variants = {
        "completeness": "random",
        "accuracy": "duration_timestamp",
        "consistency": "resource_mismatch",
        "timeliness": "event_delay",
    }

    for dimension, variant in variants.items():
        for level in DEGRADATION_LEVELS:
            configs.append(
                DegradationConfig(
                    dimension=dimension,
                    level=level,
                    variant=variant,
                    seed=degradation_seed,
                )
            )

    return configs
