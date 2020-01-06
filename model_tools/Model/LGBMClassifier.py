# -*- coding: utf-8 -*-

"""
lightgbm train sklearn-wrapper add some new func.

"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from statsmodels.distributions import ECDF
from sklearn.metrics import roc_auc_score
from ..metrics import lgb_ks, ks
import lightgbm as lgbm
import numpy as np
import pandas as pd


class LGBMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, silent=False,
                 boosting_type='gbdt', max_bin=100, max_depth=7,
                 num_leaves=64, learning_rate=0.1, min_child_samples=3,
                 subsample_for_bin=50000,
                 subsample=0.95, colsample_bytree=0.95,
                 feature_fraction=0.95, bagging_fraction=0.95, bagging_freq=3,
                 reg_alpha=0, reg_lambda=0, objective='binary', min_data=1, min_data_in_bin=1,
                 scale_pos_weight=5,
                 metric='auc', num_threads=-1, seed=512,
                 random_state=512, num_rounds=10000, early_stopping_rounds=100, feval=lgb_ks,
                 verbose_eval=False, categorical_feature=None, predict_method='cv_5'
                 ):

        self.silent = silent
        # Parameter for Tree Booster
        self.boosting_type = boosting_type
        self.max_bin = max_bin
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.subsample_for_bin = subsample_for_bin
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.scale_pos_weight = scale_pos_weight
        self.num_threads = num_threads
        self.min_data = min_data
        self.min_data_in_bin = min_data_in_bin
        # Misc
        self.metric = metric
        self.seed = seed
        # split
        self.random_state = random_state
        # earlystopping
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.feval = feval
        self.verbose_eval = verbose_eval
        # predict method
        self.predict_method = predict_method

        self.categorical_feature = categorical_feature
        self.bst = None

    def cv(self, X, y):
        dtrain = lgbm.Dataset(X, y, categorical_feature=self.categorical_feature)
        param = {
            'silent': 1 if self.silent else 0,
            'num_round': self.num_rounds,
            'num_threads': self.num_threads,
            'boosting_type': self.boosting_type,
            'learning_rate': self.learning_rate,
            'max_bin': self.max_bin,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample_for_bin': self.subsample_for_bin,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'objective': self.objective,
            'metric': self.metric,
            'seed': self.seed
        }
        results = lgbm.cv(param, dtrain, self.num_rounds, nfold=5, early_stopping_rounds=self.early_stopping_rounds,
                          feval=self.feval)
        return results

    def fit(self, X, y):
        if hasattr(X, 'reset_index'):
            X = X.reset_index(drop=True)
        param = {
            # 'num_round': self.num_rounds,
            'num_threads': self.num_threads,
            'boosting_type': self.boosting_type,
            'learning_rate': self.learning_rate,
            'max_bin': self.max_bin,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            # 'subsample_for_bin': self.subsample_for_bin,
            'subsample': self.subsample,
            #'colsample_bytree': self.colsample_bytree,
            'feature_fraction': self.feature_fraction,
            #'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            #'min_data': self.min_data,
            #'min_data_in_bin': self.min_data_in_bin,
            # 'objective': self.objective,
            'metric': self.metric,
            'verbose': -1,
            'seed': self.seed
        }

        if self.predict_method.startswith('cv'):
            
            oof_predict = np.zeros(X.shape[0])
            self.bst = []
            score_list = []
            n_folds = int(self.predict_method.split('_')[1])
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=self.seed)
            for trn_ind, val_ind in skf.split(X, y):
                X_trn, X_val = X.iloc[trn_ind, :], X.iloc[val_ind, :]
                y_trn, y_val = y.values[trn_ind], y[val_ind]
                dtrain = lgbm.Dataset(X_trn, y_trn, categorical_feature=self.categorical_feature)
                dval = lgbm.Dataset(X_val, y_val, categorical_feature=self.categorical_feature)
                bst = lgbm.train(param, dtrain, self.num_rounds, valid_sets=[dtrain, dval],
                                 valid_names=['train', 'valid'], early_stopping_rounds=100,
                                 feval=self.feval, verbose_eval=False)
                if self.metric == 'auc' and self.feval is None:
                    score_list.append(bst.best_score['valid']['auc'])
                elif self.feval.__name__.endswith('ks'):
                    score_list.append(bst.best_score['valid']['ks'])
                self.bst.append(bst)
                oof_predict[val_ind] = bst.predict(X_val)

            self.score_avg = np.mean(score_list)
            self.score_std = np.std(score_list)
            print("cross validation get score: {0} ± {1} ".format(round(np.mean(score_list), 4),
                                                                  round(np.std(score_list), 4)))
            print("train set full auc score: ", roc_auc_score(y, oof_predict))
            print("train set full ks score: ", ks(y, oof_predict))
            self.oof_predict = oof_predict
            self.ecdf = ECDF(oof_predict)

        elif self.predict_method == 'split_tune':
            X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.3, random_state=self.random_state)
            dtrain = lgbm.Dataset(X_trn, y_trn, categorical_feature=self.categorical_feature)
            dval = lgbm.Dataset(X_val, y_val, categorical_feature=self.categorical_feature)
            self.bst = lgbm.train(param, dtrain, self.num_rounds, valid_sets=[dtrain, dval],
                                  valid_names=['train', 'valid'], early_stopping_rounds=100,
                                  feval=self.feval, verbose_eval=False)
            if self.feval is not None:
                print("cross validation get score: {} ".format(round(self.bst.best_score['valid']['ks'], 4)))
            else:
                print("cross validation get score: {} ".format(round(self.bst.best_score['valid']['auc'], 4)))
        else:
            dtrain = lgbm.Dataset(X, y, categorical_feature=self.categorical_feature)
            self.bst = lgbm.train(param, dtrain, self.num_round)
        return self

    def predict(self, X):
        if self.predict_method.startswith('cv'):
            X = X[self.bst[0].feature_name()]
            for i, bst in enumerate(self.bst):
                if i == 0:
                    preds = bst.predict(X) / len(self.bst)
                else:
                    preds += bst.predict(X) / len(self.bst)
        else:
            try:
                X = X[self.bst.feature_name()]
                preds = self.bst.predict(X)
            except:
                preds = self.bst.predict(X)
        return preds

    def predict_proba(self, X):
        if self.predict_method.startswith('cv'):
            X = X[self.bst[0].feature_name()]
        else:
            try:
                X = X[self.bst.feature_name()]
            except:
                X = X.copy()
        predictions = self.predict(X)
        if self.objective == 'multiclass':
            return predictions
        return np.vstack([1 - predictions, predictions]).T

    @property
    def feature_importances_(self):
        if self.predict_method.startswith('cv'):
            feature_imp = pd.DataFrame({'variable': self.bst[0].feature_name()})
            feature_imp['importance'] = 0
            for bst in self.bst:
                feature_imp['importance'] += bst.feature_importance() / len(self.bst)
        else:
            feature_imp = pd.DataFrame({'variable': self.bst.feature_name()})
            feature_imp['importance'] = self.bst.feature_importance()
        return feature_imp.sort_values(by=['importance'], ascending=False)
