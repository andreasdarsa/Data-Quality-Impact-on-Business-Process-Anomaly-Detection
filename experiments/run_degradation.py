from src.data.loader import load_logs, save_logs
from src.data.schema import validate_logs
from src.degradation import apply_degradation
from src.process_mining import build_dfg, prune_dfg
from src.features import extract_features
from src.models import fit_all_detectors, evaluate_all_detectors
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
    SPLIT_SEEDS,
    DEGRADATION_SEEDS,
    THRESHOLD_PERCENTILE,
    get_binet_train_filename,
    get_binet_test_filename,
    get_degradation_configs,
)


def config_to_dict(config) -> dict:
    return {
        "dimension": config.dimension,
        "level": config.level,
        "variant": config.variant,
        "seed": config.seed,
    }


def build_output_name(config, split_seed: int) -> str:
    level_str = str(config.level).replace(".", "_")

    return (
        f"{config.dimension}"
        f"__{config.variant}"
        f"__level_{level_str}"
        f"__degradation_seed_{config.seed}"
        f"__split_seed_{split_seed}"
    )


def run_single_degradation_experiment(
    test_logs: list[dict],
    dfg,
    fitted_models: dict,
    split_seed: int,
    config,
) -> dict:

    # Apply degradation to a fresh copy of the original test set.
    degraded_test_logs = apply_degradation(test_logs, config)
    validate_logs(degraded_test_logs)

    output_name = build_output_name(config, split_seed)

    degraded_log_path = DEGRADED_DATA_DIR / f"{output_name}.json"
    save_logs(degraded_test_logs, degraded_log_path)

    # Extract features using the same training DFG.
    X_test, y_test, test_subtype = extract_features(
        degraded_test_logs,
        dfg
    )

    # Evaluate using the already fitted detectors.
    # No retraining or scaler refitting takes place here.
    detector_outputs = evaluate_all_detectors(
        fitted_models=fitted_models,
        X_test=X_test,
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

    print(
        f"[split_seed={split_seed}] "
        f"Completed: {output_name}"
    )

    return result


def main() -> None:

    for split_seed in SPLIT_SEEDS:

        print(
            f"\n========== Split seed: {split_seed} =========="
        )

        # 1. Load original, non-degraded BINet datasets.
        train_logs = load_logs(
            PROCESSED_DATA_DIR /
            get_binet_train_filename(split_seed)
        )

        test_logs = load_logs(
            PROCESSED_DATA_DIR /
            get_binet_test_filename(split_seed)
        )

        validate_logs(train_logs)
        validate_logs(test_logs)

        # 2. Build the DFG using normal training cases only.
        normal_train_logs = [
            case for case in train_logs
            if not case["is_anomaly"]
        ]

        dfg = build_dfg(normal_train_logs)
        dfg = prune_dfg(dfg, min_freq=2)

        # 3. Extract training features from ALL training cases.
        X_train, y_train, _ = extract_features(
            train_logs,
            dfg
        )

        # 4. Fit all detectors exactly once for this split.
        fitted_models = fit_all_detectors(
            X_train=X_train,
            threshold_percentile=THRESHOLD_PERCENTILE,
            seed=split_seed,
        )

        print(
            f"Detectors fitted for split seed {split_seed}."
        )

        # 5. Run independent degradation repetitions.
        for degradation_seed in DEGRADATION_SEEDS:

            print(
                f"\n--- Degradation repetition: "
                f"{degradation_seed} ---"
            )

            configs = get_degradation_configs(
                degradation_seed
            )

            for config in configs:
                run_single_degradation_experiment(
                    test_logs=test_logs,
                    dfg=dfg,
                    fitted_models=fitted_models,
                    split_seed=split_seed,
                    config=config,
                )

    print("\nAll degradation experiments completed.")


if __name__ == "__main__":
    main()
