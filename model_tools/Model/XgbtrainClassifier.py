# -*- coding: utf-8 -*-

"""
xgboost train sklearn-wrapper add some new func.

@author: Moon
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from ..metrics import xgb_ks
import xgboost as xgb
import numpy as np
import pandas as pd


class XGBoostClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, silent=True,
                 use_buffer=True, ntree_limit=0,
                 nthread=-1, booster='gbtree',
                 learning_rate=0.1, gamma=0.01,
                 max_depth=7, min_child_weight=3, subsample=0.95,
                 colsample_bytree=1,
                 alpha=0, lambda_bias=0, objective='binary:logistic',
                 eval_metric='auc', seed=512, num_class=None,
                 max_delta_step=0, random_state=512,
                 num_rounds=5000, early_stopping_rounds=100, feval=xgb_ks,
                 maximize=True, verbose_eval=False, predict_method='cv_5'
                 ):
        assert booster in ['gbtree', 'gblinear']
        assert objective in ['reg:linear', 'reg:logistic',
                             'binary:logistic', 'binary:logitraw', 'multi:softmax',
                             'multi:softprob', 'rank:pairwise']
        assert eval_metric in [None, 'rmse', 'mlogloss', 'logloss', 'error',
                               'merror', 'auc', 'ndcg', 'map', 'ndcg@n', 'map@n']

        self.silent = silent
        self.use_buffer = use_buffer
        self.ntree_limit = ntree_limit
        self.nthread = nthread
        self.booster = booster
        # Parameter for Tree Booster
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_delta_step = max_delta_step
        # Parameter for Linear Booster
        self.alpha = alpha
        self.lambda_bias = lambda_bias
        # Misc
        self.objective = objective
        self.eval_metric = eval_metric
        self.seed = seed
        self.num_class = num_class
        # split
        self.random_state = random_state
        # earlystopping
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.feval = feval
        self.maximize = maximize
        self.verbose_eval = verbose_eval
        # predict method
        self.predict_method = predict_method

        self.bst = None

    def cv(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        param = {
            'silent': 1 if self.silent else 0,
            'use_buffer': int(self.use_buffer),
            'num_round': self.num_rounds,
            'ntree_limit': self.ntree_limit,
            'nthread': self.nthread,
            'booster': self.booster,
            'eta': self.learning_rate,
            'gamma': self.gamma,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'max_delta_step': self.max_delta_step,
            'alpha': self.alpha,
            'lambda_bias': self.lambda_bias,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'seed': self.seed,
            'num_class': self.num_class,
        }
        results = xgb.cv(param, dtrain, self.num_rounds, 5, early_stopping_rounds=self.early_stopping_rounds,
                         feval=self.feval, maximize=self.maximize)
        return results

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        param = {
            'silent': 1 if self.silent else 0,
            'use_buffer': int(self.use_buffer),
            'num_round': self.num_rounds,
            'ntree_limit': self.ntree_limit,
            'nthread': self.nthread,
            'booster': self.booster,
            'eta': self.learning_rate,
            'gamma': self.gamma,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'max_delta_step': self.max_delta_step,
            'alpha': self.alpha,
            'lambda_bias': self.lambda_bias,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'seed': self.seed
        }
        if self.num_class is not None:
            param['num_class'] = self.num_class

        if self.predict_method.startswith('cv'):
            self.bst = []
            score_list = []
            n_folds = int(self.predict_method.split('_')[1])
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=self.seed)
            for trn_ind, val_ind in skf.split(X, y):
                X_trn, X_val = X.iloc[trn_ind, :], X.iloc[val_ind, :]
                y_trn, y_val = y.values[trn_ind], y[val_ind]
                dtrain = xgb.DMatrix(X_trn, y_trn)
                dval = xgb.DMatrix(X_val, y_val)
                watchlist = [(dtrain, 'train'), (dval, 'valid')]
                bst = xgb.train(param, dtrain, 10000, watchlist, early_stopping_rounds=100, feval=self.feval,
                                verbose_eval=False, maximize=self.maximize)
                score_list.append(bst.best_score)
                self.bst.append(bst)
            print("cross validation get score: {0} ± {1} ".format(round(np.mean(score_list), 4),
                                                                  round(np.std(score_list), 4)))
        elif self.predict_method == 'split_tune':
            X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
            dtrain = xgb.DMatrix(X_trn, y_trn)
            dval = xgb.DMatrix(X_val, y_val)
            watchlist = [(dtrain, 'train'), (dval, 'valid')]
            self.bst = xgb.train(param, dtrain, 10000, watchlist, early_stopping_rounds=100, feval=self.feval,
                                 verbose_eval=False, maximize=self.maximize)
            print("cross validation get score: {} ".format(round(np.mean(self.bst.best_score), 4)))
        else:
            dtrain = xgb.DMatrix(X, y)
            watchlist = [(dtrain, 'train')]
            self.bst = xgb.train(param, dtrain, self.num_round, watchlist)
        return self

    def predict(self, X):
        if self.predict_method.startswith('cv'):
            X = xgb.DMatrix(X[self.bst[0].feature_names])
            for i, bst in enumerate(self.bst):
                if i == 0:
                    preds = bst.predict(X) / len(self.bst)
                else:
                    preds += bst.predict(X) / len(self.bst)
        else:
            X = xgb.DMatrix(X[self.bst.feature_names])
            preds = self.bst.predict(X)
        return preds

    def predict_proba(self, X):
        if self.predict_method.startswith('cv'):
            X = xgb.DMatrix(X[self.bst[0].feature_names])
        else:
            X = xgb.DMatrix(X[self.bst[0].feature_names])
        predictions = self.predict(X)
        if self.objective == 'multi:softprob':
            return predictions
        return np.vstack([1 - predictions, predictions]).T

    @property
    def feature_importances_(self):
        if self.predict_method.startswith('cv'):
            feature_imp = pd.DataFrame(self.bst[0].feature_names, columns=['variable'])
            for i, bst in enumerate(self.bst):
                # feature_imp['importance'] = 0
                tmp = pd.DataFrame({'variable': [f_ for f_ in bst.get_fscore().keys()],
                                    'importance_{}'.format(i): np.array([v for v in bst.get_fscore().values()]) / len(
                                        self.bst)})
                feature_imp = pd.merge(feature_imp, tmp, on='variable', how='left')
            feature_imp['importance'] = feature_imp[['importance_{}'.format(i) for i in range(len(self.bst))]].\
                sum(axis=1)
        else:
            feature_imp = pd.DataFrame({'variable': [f_ for f_ in self.bst.get_fscore().keys()],
                                        'importance': [v for v in self.bst.get_fscore().values()]})
        return feature_imp[['variable', 'importance']].sort_values(by=['importance'], ascending=False)
