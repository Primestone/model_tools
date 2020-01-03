# -*- coding: utf-8 -*-

"""
catboost classifier add some new func.

@author: Moon
"""

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


class CatboostClassifierKFold(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 iterations=1000,
                 learning_rate=0.1,
                 depth=7,
                 l2_leaf_reg=40,
                 bootstrap_type='Bernoulli',
                 subsample=0.7,
                 scale_pos_weight=5,
                 eval_metric='AUC',
                 metric_period=50,
                 od_type='Iter',
                 od_wait=45,
                 random_seed=17,
                 allow_writing_files=False,
                 verbose=False,
                 cat_features_inds=None,
                 predict_method='cv_5'):

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.scale_pos_weight = scale_pos_weight
        self.eval_metric = eval_metric
        self.metric_period = metric_period
        self.od_type = od_type
        self.od_wait = od_wait
        self.random_seed = random_seed
        self.allow_writing_files = allow_writing_files
        self.verbose = verbose

        self.cat_features_inds = cat_features_inds
        # predict method
        self.predict_method = predict_method

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        self.predict_cols = list(X.columns)

        param = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'bootstrap_type': self.bootstrap_type,
            'subsample': self.subsample,
            'scale_pos_weight': self.scale_pos_weight,
            'eval_metric': self.eval_metric,
            'metric_period': self.metric_period,
            'od_type': self.od_type,
            'od_wait': self.od_wait,
            'random_seed': self.random_seed,
            'allow_writing_files': self.allow_writing_files
        }

        if self.predict_method.startswith('cv'):
            self.bst = []
            score_list = []
            n_folds = int(self.predict_method.split('_')[1])
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=self.random_seed)
            for trn_ind, val_ind in skf.split(X, y):
                X_trn, X_val = X.iloc[trn_ind, :], X.iloc[val_ind, :]
                y_trn, y_val = y.values[trn_ind], y[val_ind]
                cat_model = CatBoostClassifier(**param)
                cat_model.fit(X_trn, y_trn, eval_set=(X_val, y_val),
                              cat_features=self.cat_features_inds, use_best_model=True, verbose=self.verbose)

                self.bst.append(cat_model)
                score = roc_auc_score(y_val, cat_model.predict_proba(X_val)[:, 1])
                score_list.append(score)

            self.score_avg = np.mean(score_list)
            self.score_std = np.std(score_list)
            print("cross validation get score: {0} ± {1} ".format(round(np.mean(score_list), 4),
                                                                  round(np.std(score_list), 4)))
        elif self.predict_method == 'split_tune':
            X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
            cat_model = CatBoostClassifier(**param)
            cat_model.fit(X_trn, y_trn, eval_set=(X_val, y_val),
                          cat_features=self.cat_features_inds, use_best_model=True, verbose=self.verbose)
            self.bst = cat_model
            score = roc_auc_score(y_val, cat_model.predict_proba(X_val)[:, 1])
            print("cross validation get score: {} ".format(round(score)))
        return self

    def predict(self, X):
        if self.predict_method.startswith('cv'):
            # self.predict_cols = [str(f_, encoding='utf-8') for f_ in self.bst[0].feature_names_]
            X = X[self.predict_cols].copy()
            for i, bst in enumerate(self.bst):
                if i == 0:
                    preds = bst.predict_proba(X)[:, 1] / len(self.bst)
                else:
                    preds += bst.predict_proba(X)[:, 1] / len(self.bst)
        else:
            # self.predict_cols = [str(f_, encoding='utf-8') for f_ in self.bst.feature_names_]
            preds = self.bst.predict_proba(X[self.predict_cols])[:, 1]
        return preds

    def predict_proba(self, X):
        predictions = self.predict(X)
        return np.vstack([1 - predictions, predictions]).T

    @property
    def feature_importance_(self):
        if self.predict_method.startswith('cv'):
            feature_imp = pd.DataFrame({'variable': self.predict_cols})
            feature_imp['importance'] = 0
            for bst in self.bst:
                feature_imp['importance'] += bst.feature_importances_ / len(self.bst)
        else:
            feature_imp = pd.DataFrame({'variable': self.predict_cols})
            feature_imp['importance'] = self.bst.feature_importances_
        return feature_imp.sort_values(by=['importance'], ascending=False)

