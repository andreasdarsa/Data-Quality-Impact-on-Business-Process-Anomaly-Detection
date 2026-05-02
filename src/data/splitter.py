import random
from typing import Tuple


def get_label(case: dict) -> bool:
    return bool(case.get("is_anomaly", False))


def split_logs(
    logs: list[dict],
    test_size: float = 0.3,
    seed: int = 42,
    stratify: bool = True
) -> Tuple[list[dict], list[dict]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    rng = random.Random(seed)

    if not stratify:
        shuffled = logs[:]
        rng.shuffle(shuffled)

        n_test = int(len(shuffled) * test_size)
        return shuffled[n_test:], shuffled[:n_test]

    normal_cases = [case for case in logs if not get_label(case)]
    anomaly_cases = [case for case in logs if get_label(case)]

    train_normal, test_normal = _split_group(normal_cases, test_size, rng)
    train_anomaly, test_anomaly = _split_group(anomaly_cases, test_size, rng)

    train = train_normal + train_anomaly
    test = test_normal + test_anomaly

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


def _split_group(
    group: list[dict],
    test_size: float,
    rng: random.Random
) -> Tuple[list[dict], list[dict]]:
    shuffled = group[:]
    rng.shuffle(shuffled)

    n_test = int(len(shuffled) * test_size)

    test = shuffled[:n_test]
    train = shuffled[n_test:]

    return train, test