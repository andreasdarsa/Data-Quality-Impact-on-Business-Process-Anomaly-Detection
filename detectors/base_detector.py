"""
This module defines the BaseDetector class, which serves as an abstract base class for anomaly detection algorithms.
It provides a common interface for fitting a model to data and making predictions. 
Subclasses should implement the fit and predict methods to provide specific anomaly detection functionality.
"""


class BaseDetector:

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score_samples(self, X):
        raise NotImplementedError

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
