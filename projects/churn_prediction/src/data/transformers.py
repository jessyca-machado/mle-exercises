import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ToFloat32(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # se vier pandas:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return np.asarray(X, dtype=np.float32)

