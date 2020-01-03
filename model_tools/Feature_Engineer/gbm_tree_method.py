# -*- coding: utf-8 -*-

"""
generate GBDT tree index and One-hot encode as InputX

@author: Moon
"""
import pandas as pd
from ..utils import is_pandas
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier


class GBMEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, gbm_params, target, append=False):
        self.gbm_params = gbm_params
        self.target = target
        self.append = append
        self.predictors = None
        self.gbm = None
        self.ohenc = None

    def fit(self, X, y=None):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")

        dtypes = X.dtypes
        self.predictors = [f_ for f_ in X.columns if f_ != self.target and dtypes[f_] not in ['object', 'datetime64[ns]']]
        gbm = GradientBoostingClassifier(**self.gbm_params)
        gbm.fit(X[self.predictors], X[self.target])

        ohenc = OneHotEncoder()
        ohenc.fit(gbm.apply(X[self.predictors])[:, :, 0])
        self.gbm = gbm
        self.ohenc = ohenc
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        if not is_pandas(X):
            raise ValueError("Input x must a dataframe!")

        x = self.ohenc.transform(self.gbm.apply(X[self.predictors])[:, :, 0])
        ncols = x.shape[1]
        columns = ['tree_ind{}'.format(i) for i in range(ncols)]
        output = pd.DataFrame(data=x.toarray(), columns=columns)
        if not self.append:
            return output
        else:
            return pd.concat([X.reset_index(drop=True), output], axis=1)
