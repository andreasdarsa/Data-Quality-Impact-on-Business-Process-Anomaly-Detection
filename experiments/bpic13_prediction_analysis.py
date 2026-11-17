
import json
from pathlib import Path

import numpy as np

from src.data.loader import load_logs
from src.process_mining import build_dfg, prune_dfg
from src.features import extract_features
from src.models import fit_all_detectors, evaluate_all_detectors
from src.utils.paths import (
    PROCESSED_DATA_DIR,
    DEGRADED_DATA_DIR,
)

from experiments.configs import (
    SPLIT_SEEDS,
    DEGRADATION_SEEDS,
    THRESHOLD_PERCENTILE,
    get_bpic13_train_filename,
    get_bpic13_test_filename,
    get_degradation_configs,
)

from experiments.bpic13_degradation import build_output_name


DATASET = "bpic13"
OUTPUT_DIR = Path("results/prediction_analysis")
OUTPUT_FILE = OUTPUT_DIR / "bpic13_prediction_analysis.json"


def confusion_counts(y_true, y_pred):
    """Calculate TP, TN, FP and FN."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def analyze_predictions(
    y_true,
    y_subtypes,
    clean_pred,
    degraded_pred,
    affected_mask,
):
    """Compare clean and degraded predictions, including FP attribution."""

    y_true = np.asarray(y_true, dtype=int)
    clean_pred = np.asarray(clean_pred, dtype=int)
    degraded_pred = np.asarray(degraded_pred, dtype=int)
    affected_mask = np.asarray(affected_mask, dtype=bool)

    if not (
        len(y_true)
        == len(clean_pred)
        == len(degraded_pred)
        == len(affected_mask)
    ):
        raise ValueError(
            "Prediction, label, and affected-mask lengths do not match."
        )

    changed = clean_pred != degraded_pred
    total = len(y_true)

    normal_mask = y_true == 0
    degraded_positive = degraded_pred == 1
    newly_positive = (clean_pred == 0) & (degraded_pred == 1)

    false_positive_analysis = {
        "affected_cases": int(affected_mask.sum()),
        "unaffected_cases": int((~affected_mask).sum()),
        "fp_aff": int(np.sum(
            normal_mask & affected_mask & degraded_positive
        )),
        "fp_unaff": int(np.sum(
            normal_mask & ~affected_mask & degraded_positive
        )),
        "fp_new_aff": int(np.sum(
            normal_mask & affected_mask & newly_positive
        )),
    }

    per_subtype = {}

    for subtype in sorted(set(y_subtypes)):
        mask = np.asarray(
            [value == subtype for value in y_subtypes],
            dtype=bool,
        )

        subtype_total = int(mask.sum())
        subtype_changed = int(changed[mask].sum())

        per_subtype[subtype] = {
            "total_cases": subtype_total,
            "changed_predictions": subtype_changed,
            "change_rate": (
                float(subtype_changed / subtype_total)
                if subtype_total else 0.0
            ),
            "clean": confusion_counts(
                y_true[mask],
                clean_pred[mask],
            ),
            "degraded": confusion_counts(
                y_true[mask],
                degraded_pred[mask],
            ),
        }

    return {
        "total_cases": total,
        "changed_predictions": int(changed.sum()),
        "unchanged_predictions": int((~changed).sum()),
        "change_rate": (
            float(changed.mean()) if total else 0.0
        ),
        "clean": confusion_counts(y_true, clean_pred),
        "degraded": confusion_counts(y_true, degraded_pred),
        "false_positive_analysis": false_positive_analysis,
        "per_subtype": per_subtype,
    }



def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    total_experiments = (
        len(SPLIT_SEEDS)
        * len(DEGRADATION_SEEDS)
        * len(get_degradation_configs(DEGRADATION_SEEDS[0]))
    )

    completed = 0

    for split_seed in SPLIT_SEEDS:

        print(
            f"\n========== BPIC13 Split seed: {split_seed} =========="
        )

        # 1. Load clean training and test logs.
        train_logs = load_logs(
            PROCESSED_DATA_DIR /
            get_bpic13_train_filename(split_seed)
        )

        clean_test_logs = load_logs(
            PROCESSED_DATA_DIR /
            get_bpic13_test_filename(split_seed)
        )

        # 2. Build DFG using only normal training cases.
        normal_train_logs = [
            case for case in train_logs
            if not case["is_anomaly"]
        ]

        if not normal_train_logs:
            raise ValueError(
                f"No normal training cases for split {split_seed}."
            )

        dfg = build_dfg(normal_train_logs)
        dfg = prune_dfg(dfg, min_freq=2)

        # 3. Extract features from ALL training cases.
        X_train, _, _ = extract_features(
            train_logs,
            dfg,
            dataset=DATASET,
        )

        # 4. Fit detectors once on clean training data.
        fitted_models = fit_all_detectors(
            X_train=X_train,
            threshold_percentile=THRESHOLD_PERCENTILE,
            seed=split_seed,
        )

        # 5. Evaluate the clean test set.
        X_clean, y_clean, subtype_clean = extract_features(
            clean_test_logs,
            dfg,
            dataset=DATASET,
        )

        clean_outputs = evaluate_all_detectors(
            fitted_models=fitted_models,
            X_test=X_clean,
        )

        y_true = y_clean.to_numpy(dtype=int)
        subtypes = subtype_clean.tolist()

        print(
            f"Models fitted. Clean test cases: {len(y_true)}"
        )

        # 6. Compare against every existing degraded log.
        for degradation_seed in DEGRADATION_SEEDS:

            configs = get_degradation_configs(
                degradation_seed
            )

            for config in configs:

                experiment_name = build_output_name(
                    config,
                    split_seed,
                )

                degraded_log_path = (
                    DEGRADED_DATA_DIR /
                    f"{experiment_name}.json"
                )

                if not degraded_log_path.exists():
                    raise FileNotFoundError(
                        f"Missing degraded log: {degraded_log_path}"
                    )

                degraded_logs = load_logs(degraded_log_path)

                # 7. Extract degraded test features.
                X_degraded, y_degraded, subtype_degraded = (
                    extract_features(
                        degraded_logs,
                        dfg,
                        dataset=DATASET,
                    )
                )

                # 8. Verify that labels and case order are unchanged.
                if not np.array_equal(
                    y_true,
                    y_degraded.to_numpy(dtype=int),
                ):
                    raise ValueError(
                        f"Labels changed in {experiment_name}"
                    )

                if subtypes != subtype_degraded.tolist():
                    raise ValueError(
                        f"Subtype order changed in {experiment_name}"
                    )

                # 9. Evaluate with the SAME fitted models.
                degraded_outputs = evaluate_all_detectors(
                    fitted_models=fitted_models,
                    X_test=X_degraded,
                )

                detector_analysis = {}

                for detector_name in clean_outputs:

                    detector_analysis[detector_name] = analyze_predictions(
                        y_true=y_true,
                        y_subtypes=subtypes,
                        clean_pred=clean_outputs[detector_name]["y_pred"],
                        degraded_pred=degraded_outputs[detector_name]["y_pred"],
                        affected_mask=[
                            case.get("quality_issue") == config.dimension
                            for case in degraded_logs
                        ],
                    )

                result = {
                    "dataset": "BPIC13",
                    "split_seed": split_seed,
                    "degradation": {
                        "dimension": config.dimension,
                        "variant": config.variant,
                        "level": config.level,
                        "seed": config.seed,
                    },
                    "methodology": {
                        "training": "clean training set",
                        "training_cases": len(train_logs),
                        "dfg": "normal training cases only",
                        "models_refitted_between_tests": False,
                        "thresholds_recalculated": False,
                    },
                    "detectors": detector_analysis,
                }

                all_results.append(result)

                completed += 1

                print(
                    f"[{completed}/{total_experiments}] "
                    f"Compared: {experiment_name}"
                )

    # 10. Save ALL comparisons in one JSON file.
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nPrediction-level analysis completed.")
    print(f"Total comparisons: {completed}")
    print(f"Results saved in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
