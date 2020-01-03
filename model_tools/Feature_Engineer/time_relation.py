# -*- coding: utf-8 -*-

"""
generate time-related features

@author: Moon
"""
import pandas as pd
import numpy as np
from datetime import datetime
from ..utils import is_pandas
from sklearn.base import BaseEstimator, TransformerMixin


def get_day(x):
    if x == 'NaT':
        res = np.nan
    else:
        res = int(x.split(' ')[0])
    return res


class GentimerelatedFeaures(BaseEstimator, TransformerMixin):

    def __init__(self, columns, trans=True, cycle=False, drop=True):
        self.columns = columns
        self.trans = trans
        self.cycle = cycle
        self.drop = drop
        self.new_date_cols_ = []

    @staticmethod
    def days_fromnow(series):
        now = datetime.now().strftime("%Y-%m-%d")
        now_pd = pd.datetime(int(now.split('-')[0]), int(now.split('-')[1]), int(now.split('-')[2]))
        diff_days = (now_pd - series).astype(str).apply(get_day).values
        return diff_days

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

        if self.trans:
            if type(self.columns) == str:
                self.columns = [self.columns]

            for col in self.columns:
                X[col] = pd.to_datetime(X[col])
                X[col + '_year'] = X[col].dt.year
                X[col + '_month'] = X[col].dt.month
                X[col + '_doy'] = X[col].dt.dayofyear
                X[col + '_dow'] = X[col].dt.dayofweek
                X[col + '_hour'] = X[col].dt.hour
                X[col + '_days_fromnow'] = self.days_fromnow(X[col])
                self.new_date_cols_ += [col + _type for _type in ['_year', '_month', '_doy', '_dow', '_hour',
                                                                  '_days_fromnow']]

        if self.trans and self.cycle:
            for f_ in self.new_date_cols_:
                f_max = X[f_].max()
                X['cycos_' + f_] = X[f_].apply(lambda x: np.cos(2 * np.pi * x / (f_max + 1)))
                X['cysin_' + f_] = X[f_].apply(lambda x: np.sin(2 * np.pi * x / (f_max + 1)))

        if self.drop:
            X = X.drop(self.columns, axis=1)

        return X
