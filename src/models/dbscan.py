import numpy as np
from sklearn.cluster import DBSCAN


def run_dbscan_detector(X_train, X_test, eps=0.8, min_samples=5):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    combined = np.vstack([X_train, X_test])
    labels = model.fit_predict(combined)

    test_labels = labels[len(X_train):]

    y_pred = (test_labels == -1).astype(int)

    scores = np.where(test_labels == -1, 1.0, 0.0)

    return y_pred, scores
