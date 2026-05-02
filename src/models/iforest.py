import numpy as np
from sklearn.ensemble import IsolationForest


def run_iforest_detector(X_train, X_test, threshold_percentile=90, seed=42):
    model = IsolationForest(
        random_state=seed,
        contamination="auto"
    )

    model.fit(X_train)

    train_scores = -model.score_samples(X_train)
    test_scores = -model.score_samples(X_test)

    threshold = np.percentile(train_scores, threshold_percentile)

    y_pred = (test_scores > threshold).astype(int)

    return y_pred, test_scores
