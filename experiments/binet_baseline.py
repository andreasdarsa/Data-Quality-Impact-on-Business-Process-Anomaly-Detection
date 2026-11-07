from src.data.loader import load_logs
from src.data.schema import validate_logs
from src.process_mining import build_dfg, prune_dfg
from src.features import extract_features
from src.models import fit_all_detectors, evaluate_all_detectors
from src.evaluation import (
    evaluate_detector_results,
    evaluate_per_subtype,
    build_experiment_result,
    save_results,
)
from src.utils.paths import PROCESSED_DATA_DIR, BASELINE_RESULTS_DIR

from experiments.configs import (
    SEEDS,
    THRESHOLD_PERCENTILE,
    get_binet_train_filename,
    get_binet_test_filename,
)


def run_binet_baseline_for_seed(seed: int) -> dict:
    train_logs = load_logs(
        PROCESSED_DATA_DIR / get_binet_train_filename(seed)
    )

    test_logs = load_logs(
        PROCESSED_DATA_DIR / get_binet_test_filename(seed)
    )

    validate_logs(train_logs)
    validate_logs(test_logs)

    normal_train_logs = [
        case for case in train_logs
        if not case["is_anomaly"]
    ]

    if not normal_train_logs:
        raise ValueError(
            f"[BINet seed={seed}] No normal training cases."
        )

    # Build DFG using only normal training cases.
    dfg = build_dfg(normal_train_logs)
    dfg = prune_dfg(dfg, min_freq=2)

    # Extract features using the same DFG.
    X_train, y_train, _ = extract_features(train_logs, dfg)
    X_test, y_test, test_subtype = extract_features(test_logs, dfg)

    # Train detectors once on clean training data.
    fitted_models = fit_all_detectors(
        X_train=X_train,
        threshold_percentile=THRESHOLD_PERCENTILE,
        seed=seed,
    )

    # Evaluate on clean test data.
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
        experiment_name="binet_baseline_clean",
        config={
            "seed": seed,
            "dfg_training_cases": "normal_only",
            "min_dfg_frequency": 2,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "detector_training_data": "clean_training_set",
        },
        overall_metrics=overall_metrics,
        per_subtype_metrics=per_subtype_metrics,
    )

    # Use a new filename to preserve previous baseline results.
    output_path = (
        BASELINE_RESULTS_DIR /
        f"binet_clean_metrics_seed_{seed}.json"
    )

    save_results(result, output_path)

    print(f"[BINet seed={seed}] Clean baseline completed.")
    print(f"[BINet seed={seed}] Normal training cases: {len(normal_train_logs)}")
    print(f"[BINet seed={seed}] Train features: {X_train.shape}")
    print(f"[BINet seed={seed}] Test features: {X_test.shape}")
    print(f"[BINet seed={seed}] Results saved to: {output_path}")

    return result


def main() -> None:
    for seed in SEEDS:
        run_binet_baseline_for_seed(seed)


if __name__ == "__main__":
    main()
