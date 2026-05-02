import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def run_lof_detector(X_train, X_test, threshold_percentile=90, n_neighbors=20):
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True
    )

    model.fit(X_train)

    train_scores = -model.score_samples(X_train)
    test_scores = -model.score_samples(X_test)

    threshold = np.percentile(train_scores, threshold_percentile)

    y_pred = (test_scores > threshold).astype(int)

    return y_pred, test_scores
