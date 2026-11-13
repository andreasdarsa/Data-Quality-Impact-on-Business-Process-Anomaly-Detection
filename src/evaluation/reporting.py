from pathlib import Path
from src.data.loader import save_json


def save_results(results: dict, path: str | Path) -> None:
    save_json(results, path)


def build_experiment_result(
    experiment_name: str,
    config: dict | None,
    overall_metrics: dict,
    per_subtype_metrics: dict,
    split_seed: int | None = None,
) -> dict:
    return {
        "experiment": experiment_name,
        "split_seed": split_seed,
        "degradation_seed": (
            config.get("seed") if config else None
        ),
        "config": config,
        "overall_metrics": overall_metrics,
        "per_subtype_metrics": per_subtype_metrics,
    }
