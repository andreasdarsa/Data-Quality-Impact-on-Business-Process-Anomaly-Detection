
"""
Paired false-positive decomposition for RQ2.

Compares clean and degraded test predictions using the same detectors
fitted once on each clean training split.

Supported datasets:
    bpic12, bpic13, both

Outputs:
    results/fp_decomposition/fp_decomposition_<dataset>.csv
    results/fp_decomposition/fp_decomposition_summary_<dataset>.csv
"""

import argparse
import numpy as np
import pandas as pd

from src.data.loader import load_json, load_logs
from src.degradation import apply_degradation
from src.process_mining import build_dfg, prune_dfg
from src.features import extract_features
from src.models import fit_all_detectors, evaluate_all_detectors
from src.utils.paths import (
    PROCESSED_DATA_DIR,
    DEGRADED_DATA_DIR,
    DEGRADATION_RESULTS_DIR,
    RESULTS_DIR,
)

from experiments.configs import (
    SPLIT_SEEDS,
    DEGRADATION_SEEDS,
    THRESHOLD_PERCENTILE,
    get_binet_train_filename,
    get_binet_test_filename,
    get_bpic13_train_filename,
    get_bpic13_test_filename,
    get_degradation_configs,
)


OUTPUT_DIR = RESULTS_DIR / "fp_decomposition"

DATASETS = {
    "bpic12": {
        "name": "BPIC12",
        "train": get_binet_train_filename,
        "test": get_binet_test_filename,
        "feature_dataset": "bpic12",
        "prefix": "",
    },
    "bpic13": {
        "name": "BPIC13",
        "train": get_bpic13_train_filename,
        "test": get_bpic13_test_filename,
        "feature_dataset": "bpic13",
        "prefix": "bpic13__",
    },
}


def build_output_name(config, split_seed: int, dataset: str) -> str:
    level_str = str(config.level).replace(".", "_")
    prefix = DATASETS[dataset]["prefix"]

    return (
        f"{prefix}{config.dimension}__{config.variant}"
        f"__level_{level_str}"
        f"__degradation_seed_{config.seed}"
        f"__split_seed_{split_seed}"
    )


def affected_mask(degraded_logs: list[dict], dimension: str) -> np.ndarray:
    """Identify cases actually modified by the requested dimension."""
    return np.asarray(
        [
            case.get("quality_issue") == dimension
            for case in degraded_logs
        ],
        dtype=bool,
    )


def decompose(y, z, y0, yd) -> dict:
    """Paired FP decomposition corresponding to Section 3.3, Eqs. 2-3."""

    y, z, y0, yd = (
        np.asarray(a).astype(bool)
        for a in (y, z, y0, yd)
    )

    if not (len(y) == len(z) == len(y0) == len(yd)):
        raise ValueError(
            "Labels, affected mask and predictions must align."
        )

    normal = ~y
    fp_base = normal & y0
    fp_deg = normal & yd

    return {
        "n_test": int(len(y)),
        "n_normal": int(normal.sum()),
        "n_anomalous": int(y.sum()),
        "n_affected": int(z.sum()),
        "n_unaffected": int((~z).sum()),

        "n_normal_aff": int((normal & z).sum()),
        "n_normal_unaff": int((normal & ~z).sum()),

        "fp_base": int(fp_base.sum()),
        "fp_deg": int(fp_deg.sum()),

        # Eq. 2
        "fp_aff": int((normal & z & yd).sum()),
        "fp_unaff": int((normal & ~z & yd).sum()),

        # Eq. 3 and unaffected counterpart
        "fp_new_aff": int(
            (normal & z & ~y0 & yd).sum()
        ),
        "fp_new_unaff": int(
            (normal & ~z & ~y0 & yd).sum()
        ),

        # False alarms that disappear after degradation
        "fp_resolved_aff": int(
            (normal & z & y0 & ~yd).sum()
        ),
        "fp_resolved_unaff": int(
            (normal & ~z & y0 & ~yd).sum()
        ),

        # Denominators for newly-triggered FP rates
        "eligible_aff": int(
            (normal & z & ~y0).sum()
        ),
        "eligible_unaff": int(
            (normal & ~z & ~y0).sum()
        ),

        "fp_delta": int(fp_deg.sum() - fp_base.sum()),
        "prediction_changes": int((y0 != yd).sum()),
    }


def saved_fp_count(output_name: str, detector: str):
    """Load the degraded FP count from the saved experiment result."""
    path = DEGRADATION_RESULTS_DIR / f"{output_name}.json"

    if not path.exists():
        return None

    result = load_json(path)
    cm = result["overall_metrics"][detector]["confusion_matrix"]

    # Confusion matrix: [[TN, FP], [FN, TP]]
    return int(cm[0][1])


def extract_dataset_features(logs, dfg, dataset: str):
    return extract_features(
        logs,
        dfg,
        dataset=DATASETS[dataset]["feature_dataset"],
    )


def run_seed(
    dataset: str,
    split_seed: int,
    degradation_seeds: list[int],
) -> list[dict]:

    cfg = DATASETS[dataset]

    print(
        f"\n========== {cfg['name']}: "
        f"split seed {split_seed} =========="
    )

    train_logs = load_logs(
        PROCESSED_DATA_DIR / cfg["train"](split_seed)
    )

    clean_test_logs = load_logs(
        PROCESSED_DATA_DIR / cfg["test"](split_seed)
    )

    # DFG uses normal training cases only.
    normal_train_logs = [
        case for case in train_logs
        if not case["is_anomaly"]
    ]

    if not normal_train_logs:
        raise ValueError(
            f"[{cfg['name']}, seed={split_seed}] "
            "No normal training cases."
        )

    dfg = prune_dfg(
        build_dfg(normal_train_logs),
        min_freq=2,
    )

    # Detectors are fitted on ALL clean training cases.
    X_train, _, _ = extract_dataset_features(
        train_logs, dfg, dataset
    )

    X_clean, y_clean, _ = extract_dataset_features(
        clean_test_logs, dfg, dataset
    )

    fitted_models = fit_all_detectors(
        X_train=X_train,
        threshold_percentile=THRESHOLD_PERCENTILE,
        seed=split_seed,
    )

    clean_outputs = evaluate_all_detectors(
        fitted_models=fitted_models,
        X_test=X_clean,
    )

    y = y_clean.to_numpy(dtype=int)

    rows = []

    for degradation_seed in degradation_seeds:

        for config in get_degradation_configs(degradation_seed):

            output_name = build_output_name(
                config,
                split_seed,
                dataset,
            )

            degraded_path = (
                DEGRADED_DATA_DIR / f"{output_name}.json"
            )

            if degraded_path.exists():
                degraded_logs = load_logs(degraded_path)
                log_source = "saved"
            else:
                # Deterministic fallback; reapply in memory only.
                degraded_logs = apply_degradation(
                    clean_test_logs,
                    config,
                )
                log_source = "reapplied_in_memory"

            if len(degraded_logs) != len(clean_test_logs):
                raise ValueError(
                    f"{output_name}: case count changed."
                )

            # Verify case order using the available identifiers.
            clean_ids = [
                case.get("case_id", case.get("id"))
                for case in clean_test_logs
            ]

            degraded_ids = [
                case.get("case_id", case.get("id"))
                for case in degraded_logs
            ]

            if clean_ids != degraded_ids:
                raise ValueError(
                    f"{output_name}: case IDs/order differ."
                )

            X_degraded, y_degraded, _ = extract_dataset_features(
                degraded_logs,
                dfg,
                dataset,
            )

            if list(X_clean.columns) != list(X_degraded.columns):
                raise ValueError(
                    f"{output_name}: feature columns differ."
                )

            if not np.array_equal(
                y,
                y_degraded.to_numpy(dtype=int),
            ):
                raise ValueError(
                    f"{output_name}: labels/order changed."
                )

            z = affected_mask(
                degraded_logs,
                config.dimension,
            )

            # Same fitted models, scaler and thresholds.
            degraded_outputs = evaluate_all_detectors(
                fitted_models=fitted_models,
                X_test=X_degraded,
            )

            for detector in clean_outputs:

                metrics = decompose(
                    y,
                    z,
                    clean_outputs[detector]["y_pred"],
                    degraded_outputs[detector]["y_pred"],
                )

                saved = saved_fp_count(
                    output_name,
                    detector,
                )

                rows.append({
                    "dataset": dataset,
                    "split_seed": split_seed,
                    "dimension": config.dimension,
                    "variant": config.variant,
                    "level": config.level,
                    "degradation_seed": config.seed,
                    "detector": detector,
                    "degraded_log_source": log_source,
                    **metrics,
                    "saved_fp_deg": saved,
                    "matches_saved_run": (
                        None
                        if saved is None
                        else saved == metrics["fp_deg"]
                    ),
                })

            print(
                f"[{cfg['name']}, seed={split_seed}] "
                f"{output_name}"
            )

    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Pool counts first, then calculate rates from pooled denominators."""

    keys = [
        "dataset",
        "dimension",
        "level",
        "detector",
    ]

    count_cols = [
        "n_test",
        "n_normal",
        "n_anomalous",
        "n_affected",
        "n_unaffected",
        "fp_base",
        "fp_deg",
        "fp_aff",
        "fp_unaff",
        "fp_new_aff",
        "fp_new_unaff",
        "fp_resolved_aff",
        "fp_resolved_unaff",
        "eligible_aff",
        "eligible_unaff",
        "prediction_changes",
    ]

    summary = (
        df.groupby(keys, dropna=False)[count_cols]
        .sum()
        .reset_index()
    )

    def ratio(num, den):
        return np.divide(
            num.to_numpy(dtype=float),
            den.to_numpy(dtype=float),
            out=np.full(len(summary), np.nan),
            where=den.to_numpy(dtype=float) != 0,
        )

    summary["fpr_clean"] = ratio(
        summary["fp_base"],
        summary["n_normal"],
    )

    summary["fpr_degraded"] = ratio(
        summary["fp_deg"],
        summary["n_normal"],
    )

    summary["fpr_delta"] = (
        summary["fpr_degraded"] - summary["fpr_clean"]
    )

    summary["new_fa_rate_aff"] = ratio(
        summary["fp_new_aff"],
        summary["eligible_aff"],
    )

    summary["new_fa_rate_unaff"] = ratio(
        summary["fp_new_unaff"],
        summary["eligible_unaff"],
    )

    summary["share_new_fp_on_aff"] = ratio(
        summary["fp_new_aff"],
        summary["fp_new_aff"] + summary["fp_new_unaff"],
    )

    summary["affected_case_rate"] = ratio(
        summary["n_affected"],
        summary["n_test"],
    )

    return summary


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Paired false-positive decomposition for RQ2."
    )

    parser.add_argument(
        "--dataset",
        choices=["bpic12", "bpic13", "both"],
        default="both",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=SPLIT_SEEDS,
    )

    parser.add_argument(
        "--degradation-seeds",
        type=int,
        nargs="*",
        default=DEGRADATION_SEEDS,
    )

    args = parser.parse_args()

    datasets = (
        ["bpic12", "bpic13"]
        if args.dataset == "both"
        else [args.dataset]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset in datasets:

        rows = [
            row
            for seed in args.seeds
            for row in run_seed(
                dataset,
                seed,
                args.degradation_seeds,
            )
        ]

        if not rows:
            print(f"No rows generated for {dataset}.")
            continue

        df = pd.DataFrame(rows)

        raw_path = (
            OUTPUT_DIR / f"fp_decomposition_{dataset}.csv"
        )

        summary_path = (
            OUTPUT_DIR / f"fp_decomposition_summary_{dataset}.csv"
        )

        df.to_csv(raw_path, index=False)

        summary = summarise(df)
        summary.to_csv(summary_path, index=False)

        print(f"\n[{dataset}] Summary:")
        print(summary.round(4).to_string(index=False))

        print(f"Raw results: {raw_path}")
        print(f"Summary: {summary_path}")

        checked = df["matches_saved_run"].dropna()

        if len(checked):
            print(
                "Saved-run FP check: "
                f"{int(checked.sum())}/{len(checked)} rows match."
            )


if __name__ == "__main__":
    main()
