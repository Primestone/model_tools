# -*- coding: utf-8 -*-

"""
Prepeocessing Encoders

@author: Moon
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from statsmodels.distributions import ECDF
from ..utils import is_pandas, is_numpy
from ..estimators import LikelihoodEstimator


class CountEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, min_count=0, nan_value=-1, copy=True):
        self.categorical_features = categorical_features
        self.min_count = min_count
        self.nan_value = nan_value
        self.copy = copy
        self.counts = {}

    def fit(self, X, y=0):
        self.counts = {}
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        x = X[self.categorical_features].fillna('na')
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        for i in self.categorical_features:
            cnt = x.loc[:, i].value_counts().to_dict()
            if self.min_count > 0:
                cnt = dict((k, self.nan_value if v < self.min_count else v) for k, v in cnt.items())
            self.counts.update({i: cnt})
        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        inner_categorical = [x for x in X.columns if x in self.categorical_features]
        x = X[inner_categorical].fillna('na')

        if self.copy:
            x = X[inner_categorical].fillna('na').copy()

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        for i in inner_categorical:
            cnt = self.counts[i]
            x.loc[:, i] = x.loc[:, i].map(cnt)
        x = x.add_prefix('cnt_enc_')
        output = pd.concat([X, x], axis=1)
        return output


class LikelihoodEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, input_cols, alpha=0, leave_one_out=False, noise=0, seed=0):
        self.input_cols = input_cols
        self.alpha = alpha
        self.leave_one_out = leave_one_out
        self.noise = noise
        self.seed = seed
        self.nclass = None
        self.estimators = []

    def fit(self, X, y):
        x = X[self.input_cols]

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        if not is_numpy(x):
            x = np.array(x)

        self.nclass = np.unique(y).shape[0]

        for i in range(ncols):
            self.estimators.append(LikelihoodEstimator(seed=self.seed, alpha=self.alpha, noise=self.noise,
                                                       leave_one_out=self.leave_one_out).fit(x[:, i], y))
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        x = X[self.input_cols]
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]
        if not is_numpy(x):
            x = np.array(x)

        likelihoods = None

        for i in range(ncols):
            lh = self.estimators[i].predict(x[:, i], noise=True).reshape(-1, 1)
            likelihoods = np.hstack((lh,)) if likelihoods is None else np.hstack((likelihoods, lh))

        if type(self.input_cols) == list:
            lh_cols = ['lh_enc_'+'__'.join(self.input_cols)]
        elif type(self.input_cols) == str:
            lh_cols = ['lh_enc_'+self.input_cols]

        likelihoods = pd.DataFrame(data=likelihoods, columns=lh_cols)
        output = pd.concat([X.reset_index(drop=True), likelihoods], axis=1)
        return output


class PercentileEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, input_cols, apply_ppf=False, copy=True):
        self.input_cols = input_cols
        self.ppf = lambda x: norm.ppf(x * .998 + .001) if apply_ppf else x
        self.copy = copy
        self.ecdfs = {}

    def fit(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        x = X[self.input_cols]
        self.ecdfs = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        for i in range(ncols):
            self.ecdfs.update({i: ECDF(x.iloc[:, i].values)})
        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        x = X[self.input_cols]
        if self.copy:
            x = X[self.input_cols].copy()

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        for i in range(ncols):
            ecdf = self.ecdfs[i]
            x.iloc[:, i] = self.ppf(ecdf(x.iloc[:, i]))
        x = x.add_prefix('perc_enc_')
        output = pd.concat([X, x], axis=1)
        return output


class InfrequentValueEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, threshold=10, value=-1, copy=True):
        self.categorical_features = categorical_features
        self.threshold = threshold
        self.value = value
        self.copy = copy
        self.new_values = {}

    def fit(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        x = X[self.categorical_features]
        self.new_values = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        for i in range(ncols):
            val = x.iloc[:, i].value_counts().to_dict()
            val = dict((k, self.value if v < self.threshold else k) for k, v in val.items())
            self.new_values.update({i: val})
        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        x = X[self.categorical_features]
        if self.copy:
            x = X[self.categorical_features].copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        for i in range(ncols):
            val = self.new_values[i]
            x.iloc[:, i] = x.iloc[:, i].map(val)
        return x


class CategoryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, min_count=0, first_category=1, copy=True):
        self.categorical_features = categorical_features
        self.min_count = min_count
        self.first_category = first_category
        self.copy = copy
        self.encoders = {}
        self.ive = None

    def fit(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")

        x = X[self.categorical_features].fillna('na')
        self.encoders = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        if self.min_count > 0:
            self.ive = InfrequentValueEncoder(threshold=self.min_count, value=np.finfo(float).min)
            x = self.ive.fit_transform(x)

        for i in self.categorical_features:
            try:
                enc = LabelEncoder().fit(x.loc[:, i])
                self.encoders.update({i: enc})
            except:
                ind_v = x.loc[:, i].drop_duplicates().fillna('na').reset_index(drop=True).to_dict()
                v_ind = {v: k for k, v in ind_v.items()}
                self.encoders.update({i: v_ind})

        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")

        inner_categorical = [x for x in X.columns if x in self.categorical_features]
        x = X[inner_categorical].fillna('na')
        if self.copy:
            x = X[inner_categorical].fillna('na').copy()

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        ncols = x.shape[1]

        if self.ive is not None:
            x = self.ive.transform(x)

        for i in inner_categorical:
            enc = self.encoders[i]
            if hasattr(enc, 'transform'):
                x.loc[:, i] = enc.transform(x.loc[:, i]) + self.first_category
            else:
                x.loc[:, i] = x.loc[:, i].fillna('na').map(enc)
        X[inner_categorical] = x.values

        return X


class NaEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, fill_value=-999):
        self.fill_value = fill_value

    def fit(self, X):
        return self

    def transform(self, X):
        return X.fillna(self.fill_value)


class Scaler(BaseEstimator, TransformerMixin):

    def __init__(self, target, method=MinMaxScaler(), keep_catcols=True, categorical_features=None):
        self.target = target
        self.method = method
        self.keep_catcols = keep_catcols
        self.categorical_features = categorical_features

    def fit(self, X, y=0):
        predictors = [f_ for f_ in X.columns if f_ != self.target and f_ not in self.categorical_features]
        self.method.fit(X[predictors])
        self.predictors = predictors
        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X)

    def transform(self, X):
        data = self.method.transform(X[self.predictors])
        output = pd.concat([pd.DataFrame(data=data, columns=self.predictors), X[self.target]], axis=1)
        if self.keep_catcols and self.categorical_features is not None:
            try:
                output[self.categorical_features] = X[self.categorical_features].astype(int).values
            except:
                output[self.categorical_features] = X[self.categorical_features]
        return output


class DummyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns, nan_as_category=True):
        self.columns = columns
        self.nan_as_category = nan_as_category

    def fit(self, X, y=None):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")
        for col in self.columns:
            if X[col].value_counts().shape[0] > 2:
                tmp = pd.get_dummies(X[col], dummy_na=self.nan_as_category, prefix=col)
                X = pd.concat([X.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)
        return X
