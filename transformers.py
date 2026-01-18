from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.cols, errors='ignore')


class SafeTypeEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mapping_ = {k: i for i, k in enumerate(X['type'].unique())}
        self.unknown_value_ = len(self.mapping_)
        return self

    def transform(self, X):
        X = X.copy()
        X['type'] = X['type'].map(
            lambda x: self.mapping_.get(x, self.unknown_value_)
        )
        return X


class PowerTransformColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.pt = PowerTransformer(method='yeo-johnson', standardize=True)

    def fit(self, X, y=None):
        self.pt.fit(X[self.cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = self.pt.transform(X[self.cols])
        return X


class ScaleColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = self.scaler.transform(X[self.cols])
        return X
