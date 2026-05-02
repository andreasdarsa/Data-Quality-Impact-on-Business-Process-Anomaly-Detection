from dataclasses import dataclass


VALID_DIMENSIONS = {
    "completeness",
    "accuracy",
    "consistency",
    "timeliness",
}


@dataclass(frozen=True)
class DegradationConfig:
    dimension: str
    level: float
    variant: str = "default"
    seed: int = 42

    def __post_init__(self):
        if self.dimension not in VALID_DIMENSIONS:
            raise ValueError(
                f"Invalid degradation dimension: {self.dimension}"
            )

        if not 0 <= self.level <= 1:
            raise ValueError("level must be between 0 and 1.")
