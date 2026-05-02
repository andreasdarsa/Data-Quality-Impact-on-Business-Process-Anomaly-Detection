from .config import DegradationConfig
from .completeness import inject_missing_events
from .accuracy import inject_accuracy_noise
from .consistency import inject_consistency_issues
from .timeliness import inject_timeliness_issues


def apply_degradation(
    logs: list[dict],
    config: DegradationConfig
) -> list[dict]:
    if config.dimension == "completeness":
        return inject_missing_events(
            logs=logs,
            level=config.level,
            variant=config.variant,
            seed=config.seed,
        )

    if config.dimension == "accuracy":
        return inject_accuracy_noise(
            logs=logs,
            level=config.level,
            variant=config.variant,
            seed=config.seed,
        )

    if config.dimension == "consistency":
        return inject_consistency_issues(
            logs=logs,
            level=config.level,
            variant=config.variant,
            seed=config.seed,
        )

    if config.dimension == "timeliness":
        return inject_timeliness_issues(
            logs=logs,
            level=config.level,
            variant=config.variant,
            seed=config.seed,
        )

    raise ValueError(f"Unknown degradation dimension: {config.dimension}")
