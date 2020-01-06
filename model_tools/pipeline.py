# -*- coding: utf-8 -*-

"""
pipeline tools
"""

from .utils import is_pandas
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")

        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")

        return X[self.columns]


class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        assert type(columns) == list
        self.columns = columns

    def fit(self, X, y=None):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")

        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")
        X = X.drop(self.columns, axis=1)
        return X


class ConstantDropper(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=1e-10):
        self.threshold = threshold
        self.sel_cols = None

    def fit(self, X, y=None):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")

        stat_std = X.std()
        sel_cols = stat_std[stat_std >= self.threshold].index.tolist()
        self.sel_cols = sel_cols
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X)

    def transform(self, X):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")

        return X[self.sel_cols]

