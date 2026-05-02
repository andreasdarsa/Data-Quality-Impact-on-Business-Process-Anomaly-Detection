from src.data.loader import load_logs, save_logs
from src.data.schema import validate_logs
from src.degradation import apply_degradation
from src.process_mining import build_dfg, prune_dfg
from src.features import extract_features
from src.models import run_all_detectors
from src.evaluation import (
    evaluate_detector_results,
    evaluate_per_subtype,
    build_experiment_result,
    save_results,
)
from src.utils.paths import (
    PROCESSED_DATA_DIR,
    DEGRADED_DATA_DIR,
    DEGRADATION_RESULTS_DIR,
)

from experiments.configs import (
    SEEDS,
    THRESHOLD_PERCENTILE,
    get_train_filename,
    get_test_filename,
    get_degradation_configs,
)


def config_to_dict(config) -> dict:
    return {
        "dimension": config.dimension,
        "level": config.level,
        "variant": config.variant,
        "seed": config.seed,
    }


def build_output_name(config) -> str:
    level_str = str(config.level).replace(".", "_")

    return (
        f"{config.dimension}"
        f"__{config.variant}"
        f"__level_{level_str}"
        f"__seed_{config.seed}"
    )


def run_single_degradation_experiment(seed: int, config) -> dict:
    train_logs = load_logs(PROCESSED_DATA_DIR / get_train_filename(seed))
    test_logs = load_logs(PROCESSED_DATA_DIR / get_test_filename(seed))

    validate_logs(train_logs)
    validate_logs(test_logs)

    degraded_test_logs = apply_degradation(test_logs, config)
    validate_logs(degraded_test_logs)

    output_name = build_output_name(config)

    degraded_log_path = DEGRADED_DATA_DIR / f"{output_name}.json"
    save_logs(degraded_test_logs, degraded_log_path)

    dfg = build_dfg(train_logs)
    dfg = prune_dfg(dfg, min_freq=2)

    X_train, y_train, _ = extract_features(train_logs, dfg)
    X_test, y_test, test_subtype = extract_features(degraded_test_logs, dfg)

    detector_outputs = run_all_detectors(
        X_train=X_train,
        X_test=X_test,
        threshold_percentile=THRESHOLD_PERCENTILE,
        seed=seed,
    )

    overall_metrics = evaluate_detector_results(
        y_true=y_test,
        detector_results=detector_outputs,
    )

    per_subtype_metrics = evaluate_per_subtype(
        y_true=y_test,
        subtype=test_subtype,
        detector_results=detector_outputs,
    )

    result = build_experiment_result(
        experiment_name="degradation",
        config=config_to_dict(config),
        overall_metrics=overall_metrics,
        per_subtype_metrics=per_subtype_metrics,
    )

    result_path = DEGRADATION_RESULTS_DIR / f"{output_name}.json"
    save_results(result, result_path)

    print(f"[seed={seed}] Completed: {output_name}")

    return result


def main() -> None:
    for seed in SEEDS:
        configs = get_degradation_configs(seed)

        for config in configs:
            run_single_degradation_experiment(seed, config)

    print("All degradation experiments completed.")


if __name__ == "__main__":
    main()
