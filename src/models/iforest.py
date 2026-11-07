import numpy as np
from sklearn.ensemble import IsolationForest


def fit_iforest_detector(
    X_train,
    threshold_percentile=90,
    seed=42
):
    model = IsolationForest(
        random_state=seed,
        contamination="auto"
    )

    model.fit(X_train)

    train_scores = -model.score_samples(X_train)

    threshold = np.percentile(
        train_scores,
        threshold_percentile
    )

    return {
        "model": model,
        "threshold": threshold,
    }


def evaluate_iforest_detector(fitted_model, X_test):
    model = fitted_model["model"]
    threshold = fitted_model["threshold"]

    test_scores = -model.score_samples(X_test)

    y_pred = (test_scores > threshold).astype(int)

    return y_pred, test_scores
