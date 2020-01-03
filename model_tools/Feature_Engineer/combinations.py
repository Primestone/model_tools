# -*- coding: utf-8 -*-

"""
categorical features combinations

@author: Moon
"""

import itertools

from ..utils import is_pandas
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class FeatureCombiner(BaseEstimator, TransformerMixin):

    def __init__(self, columns, orders=[2, 3], separator='__'):
        """
        combine categorical features.

        Parameter:
        ----------
        columns: List of columns to be combined.
        orders: Orders to which columns should be combined.
        separator: Separator to use to combined the column names and values.
        """
        self.columns = columns
        self.orders = orders
        self.separator = separator

    def fit(self, X, y=None):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        dtypes = X.dtypes
        if self.columns is None:
            self.columns = [col for col in X.columns if dtypes[col] in ('object', 'category')]

        self.new_column_names_ = []
        return self

    def transform(self, X, y=None):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        for order in self.orders:
            for combo in itertools.combinations(self.columns, order):
                col_name = self.separator.join(combo)
                self.new_column_names_.append(col_name)
                X[col_name] = X[combo[0]].apply(str).str.cat([
                    X[col].apply(str)
                    for col in combo[1:]
                ], sep=self.separator).astype('object')

        return X

