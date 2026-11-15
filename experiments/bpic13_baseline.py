
from src.data import load_logs, validate_logs
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
    BASELINE_RESULTS_DIR,
)

from experiments.configs import (
    SPLIT_SEEDS,
    THRESHOLD_PERCENTILE,
    get_bpic13_train_filename,
    get_bpic13_test_filename,
)


def run_bpic13_baseline_for_seed(seed: int) -> dict:
    # Load train/test splits.
    train_logs = load_logs(
        PROCESSED_DATA_DIR / get_bpic13_train_filename(seed)
    )

    test_logs = load_logs(
        PROCESSED_DATA_DIR / get_bpic13_test_filename(seed)
    )

    validate_logs(train_logs)
    validate_logs(test_logs)

    # Build DFG using only normal training cases.
    normal_train_logs = [
        case for case in train_logs
        if not case["is_anomaly"]
    ]

    if not normal_train_logs:
        raise ValueError(
            f"[BPIC13 seed={seed}] No normal training cases."
        )

    dfg = build_dfg(normal_train_logs)
    dfg = prune_dfg(dfg, min_freq=2)

    # Extract BPIC13-specific features.
    X_train, y_train, _ = extract_features(
        train_logs,
        dfg,
        dataset="bpic13",
    )

    X_test, y_test, test_subtype = extract_features(
        test_logs,
        dfg,
        dataset="bpic13",
    )

    # Ensure train/test feature compatibility.
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            f"[BPIC13 seed={seed}] "
            "Train/test feature columns do not match."
        )

    # Fit detectors on the training set.
    fitted_models = fit_all_detectors(
        X_train=X_train,
        threshold_percentile=THRESHOLD_PERCENTILE,
        seed=seed,
    )

    # Evaluate on the test set.
    detector_outputs = evaluate_all_detectors(
        fitted_models=fitted_models,
        X_test=X_test,
    )

    # Calculate evaluation metrics.
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
        experiment_name="bpic13_baseline_normal_only",
        config={
            "dataset": "BPIC13",
            "seed": seed,
            "dfg_training_cases": "normal_only",
            "min_dfg_frequency": 2,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "detector_training_data": "normal_only",
            "feature_extraction": "bpic13",
        },
        overall_metrics=overall_metrics,
        per_subtype_metrics=per_subtype_metrics,
        split_seed=seed,
    )

    # Save results.
    output_path = (
        BASELINE_RESULTS_DIR /
        f"bpic13_normal_only_metrics_seed_{seed}.json"
    )

    save_results(result, output_path)

    print(f"\n[BPIC13 seed={seed}] Baseline completed.")
    print(
        f"[BPIC13 seed={seed}] "
        f"Normal training cases: {len(normal_train_logs)}"
    )
    print(
        f"[BPIC13 seed={seed}] "
        f"Train features: {X_train.shape}"
    )
    print(
        f"[BPIC13 seed={seed}] "
        f"Test features: {X_test.shape}"
    )
    print(
        f"[BPIC13 seed={seed}] "
        f"Results saved to: {output_path}"
    )

    return result


def main() -> None:
    for seed in SPLIT_SEEDS:
        run_bpic13_baseline_for_seed(seed)

    print("\nAll BPIC13 baseline experiments completed.")


if __name__ == "__main__":
    main()
