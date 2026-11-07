from sklearn.preprocessing import StandardScaler

from .knn import fit_knn_detector, evaluate_knn_detector
from .lof import fit_lof_detector, evaluate_lof_detector
from .iforest import fit_iforest_detector, evaluate_iforest_detector
from .dbscan import fit_dbscan_detector, evaluate_dbscan_detector


def fit_all_detectors(
    X_train,
    threshold_percentile=90,
    seed=42
):
    """
    Scale training data, fit all detectors and calculate thresholds.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    fitted_models = {
        "scaler": scaler,

        "knn": fit_knn_detector(
            X_train_scaled,
            threshold_percentile=threshold_percentile
        ),

        "lof": fit_lof_detector(
            X_train_scaled,
            threshold_percentile=threshold_percentile
        ),

        "iforest": fit_iforest_detector(
            X_train_scaled,
            threshold_percentile=threshold_percentile,
            seed=seed
        ),

        "dbscan": fit_dbscan_detector(
            X_train_scaled,
            eps=0.8,
            min_samples=5
        ),
    }

    return fitted_models


def evaluate_all_detectors(fitted_models, X_test):
    """
    Evaluate pre-trained detectors on a test dataset.
    Uses the same scaler, models and thresholds from training.
    """

    scaler = fitted_models["scaler"]
    X_test_scaled = scaler.transform(X_test)

    results = {}

    y_pred, scores = evaluate_knn_detector(
        fitted_models["knn"],
        X_test_scaled
    )
    results["knn"] = {
        "y_pred": y_pred,
        "scores": scores
    }

    y_pred, scores = evaluate_lof_detector(
        fitted_models["lof"],
        X_test_scaled
    )
    results["lof"] = {
        "y_pred": y_pred,
        "scores": scores
    }

    y_pred, scores = evaluate_iforest_detector(
        fitted_models["iforest"],
        X_test_scaled
    )
    results["iforest"] = {
        "y_pred": y_pred,
        "scores": scores
    }

    y_pred, scores = evaluate_dbscan_detector(
        fitted_models["dbscan"],
        X_test_scaled
    )
    results["dbscan"] = {
        "y_pred": y_pred,
        "scores": scores
    }

    return results
