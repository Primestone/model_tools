# -*- coding: utf-8 -*-

"""
feature selector base on greedy search algorithm

@author: Moon
"""

import pandas as pd
import lightgbm as lgbm
from ..metrics import lgb_ks
from sklearn.base import BaseEstimator, TransformerMixin
from ..Model.LGBMClassifier import LGBMClassifier
from sklearn.model_selection import train_test_split


class GreedyFeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, target, selector=None, feval=lgb_ks, good_features=[], verbose=True,
                 params=None, method='select_loop', random_state=1024):
        self.target = target
        self.selector = selector
        self.params = params
        self.feval = feval
        self.good_features = good_features
        self._verbose = verbose
        self.method = method
        self.random_state = random_state

        self.select_features = None
        self.columns = None

    def evaluate_score(self, x1, y1):
        lgb_params = {
            'boosting_type': 'gbdt',
            'num_leaves': 2 ** 5,
            'max_depth': 5,
            'max_bin': 100,
            'min_child_samples': 100,
            'learning_rate': 0.1,
            # 'feature_fraction': 0.9,
            # 'bagging_fraction': 0.9,
            'num_threads': -1,
            # 'bagging_freq': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'seed': 99} if self.params is None else self.params

        lgbm_model = LGBMClassifier(**lgb_params)
        lgbm_model.fit(x1, y1)
        score_avg = lgbm_model.score_avg
        score_std = lgbm_model.score_st
        score = score_avg - score_std/2.0
        return round(score, 6)

    def selectionloop(self, trn_x, trn_y):
        score_history = []
        good_features = self.good_features

        loop = 1
        while (len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]) or \
                (len(score_history) >= 2 and (score_history[-1][0] > score_history[-2][0] or
                                              score_history[-1][0] > score_history[-3][0] or
                                              score_history[-1][0] > score_history[-4][0])):
            scores = []
            for feature in self.columns:
                if feature not in good_features:
                    selected_features = good_features + [feature]
                    trn_X = pd.concat([trn_x.loc[:, j] for j in selected_features], axis=1)
                    # vld_X = pd.concat([vld_x.loc[:, j] for j in selected_features], axis=1)

                    score = self.evaluate_score(trn_X, trn_y)
                    scores.append((score, feature))

            good_features.append(sorted(scores)[-1][1])
            score_history.append(sorted(scores)[-1])
            loop += 1
            if self._verbose:
                print("Current features {0} score: {1} ".format(list(good_features), score_history[-1][0]))

        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = list(good_features)
        n_features = len(good_features)
        if self._verbose:
            print("Selected features : ", good_features[:(n_features-2)])

        return good_features[:(n_features-2)]

    def deleteloop(self, trn_x, trn_y, vld_x, vld_y):
        score_history = []
        full_features = self.columns
        while (len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]) or \
                (len(score_history) >= 2 and (score_history[-1][0] > score_history[-2][0] or
                                              score_history[-1][0] > score_history[-3][0])):
            scores = []

            if len(score_history) < 1:
                features_to_use = full_features
                trn_X = trn_x[features_to_use]
                # vld_X = vld_x[features_to_use]

                score = self.evaluate_score(trn_X, trn_y)
                score_history.append((score, 'All'))

                if self._verbose:
                    print("Current score : {} ({})".format(round(score, 6), len(features_to_use)))

            else:
                for i, feature in enumerate(full_features):
                    if feature not in self.good_features:
                        features_to_use = [x for x in full_features if x != feature]
                        trn_X = trn_x[features_to_use]
                        # vld_X = vld_x[features_to_use]

                        score = self.evaluate_score(trn_X, trn_y)
                        scores.append((score, feature))

                        if self._verbose:
                            print("Current score : {0} ({1}/{2})".format(round(score, 6), i + 1,
                                                                         len(features_to_use)))

                full_features.remove(sorted(scores)[-1][1])
                score_history.append(sorted(scores)[-1])
                print(score_history)
                if self._verbose:
                    print("Current drop feature : ", sorted(scores)[-1][1])

        # Remove last added feature
        full_features.append(score_history[-1][1])
        if self._verbose:
            print("Selected features : ", full_features)
        return full_features

    def fit(self, X, y=None):
        self.columns = [f_ for f_ in X.columns if f_ != self.target]
        y = X[self.target]
        X = X[self.columns].copy()
        # trn_x, vld_x, trn_y, vld_y = train_test_split(X[self.columns], X[self.target], test_size=0.3,
        #                                              random_state=self.random_state)

        if self.method == 'select_loop':
            sel_cols = self.selectionloop(X, y)
        else:
            sel_cols = self.deleteloop(X, y)
        self.select_features = sel_cols
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        return X[self.select_features + [self.target]]


class FeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, target, estimator=LGBMClassifier, method='SelectFromModel', threshold=1, params=None):
        self.target = target
        self.estimator = estimator
        self.method = method
        self.threshold = threshold
        self.params = params

        self.sel_cols = None

    def fit(self, X, y=None):

        if self.method == 'SelectFromModel' and self.estimator is not None:
            estimator = self.estimator(**self.params)
            predictors = [f_ for f_ in X.columns if f_ != self.target]
            estimator.fit(X[predictors], X[self.target])
            feature_imp = estimator.feature_importances_
            sel_cols = feature_imp[feature_imp['importance'] >= self.threshold]['variable'].values.tolist()

        elif self.method == 'GreedySelect':
            selector = GreedyFeatureSelection(target=self.target, params=self.params, method='select_loop')
            selector.fit(X, y=None)
            sel_cols = selector.select_features

        elif self.method == 'GreedyDelete':
            selector = GreedyFeatureSelection(target=self.target, params=self.params, method='delete_loop')
            selector.fit(X, y=None)
            sel_cols = selector.select_features

        self.sel_cols = sel_cols

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        return X[self.sel_cols + [self.target]]
