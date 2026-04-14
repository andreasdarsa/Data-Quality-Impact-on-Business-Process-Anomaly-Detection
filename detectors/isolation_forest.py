from sklearn.ensemble import IsolationForest
from detectors.base_detector import BaseDetector


class IsolationForestDetector(BaseDetector):

    def __init__(self, n_estimators=100, contamination=0.1):
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        preds = self.model.predict(X)
        y_pred = (preds == -1).astype(int)
        return y_pred

    def score_samples(self, X):
        return -self.model.decision_function(X)
