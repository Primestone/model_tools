# -*- coding: utf-8 -*-
"""
Parameters Tuned base on Bayesian Optim.
"""

import gc
import numpy as np
import xgboost as xgb
from functools import partial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from bayes_opt import BayesianOptimization


def evaluator(train_df, target, **params):
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['min_child_samples'] = int(params['min_child_samples'])

    clf = LGBMClassifier(**params, n_estimators=10000, nthread=4)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
    oof_predict = np.zeros(train_df.shape[0])

    feats = [f for f in train_df.columns if f not in [target]]

    for n_fold, (train_idx, valid_idx) in enumerate(
            folds.split(train_df[feats], train_df[target])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]

        clf.fit(
            train_x,
            train_y,
            eval_set=[
                (train_x,
                 train_y),
                (valid_x,
                 valid_y)],
            eval_metric='auc',
            verbose=False,
            early_stopping_rounds=100)

        oof_predict[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_)[:, 1]

        del train_x, train_y, valid_x, valid_y
        gc.collect()

    return roc_auc_score(train_df[target], oof_predict)


def xgb_evaluator(train_df, target, **params):
    params['max_depth'] = int(params['max_depth'])
    params['min_child_samples'] = int(params['min_child_samples'])
    params['objective'] = 'binary:logistic'
    params['n_estimators'] = 1000
    params['seed'] = 42

    feats = [f for f in train_df.columns if f not in [target]]

    xgb_train = xgb.DMatrix(
        train_df[feats], label=train_df[target])

    cv_result = xgb.cv(
        params,
        xgb_train,
        num_boost_round=10000,
        nfold=5,
        metrics='auc',
        seed=42,
        early_stopping_rounds=50,
        verbose_eval=False)

    num_round_best = cv_result.shape[0] + 1
    params['n_estimators'] = num_round_best
    auc_mean = round(max(cv_result['test-auc-mean']), 4)
    return auc_mean


def BayesOptim(train_df, target, init_points=5, n_iter=15):
    params = {
        'colsample_bytree': (0.8, 1),
        'learning_rate': (.01, .14),
        'num_leaves': (16, 64),
        'subsample': (0.8, 1),
        'max_depth': (4, 7),
        'reg_alpha': (.03, .5),
        'reg_lambda': (.06, .5),
        'min_split_gain': (.01, .03),
        # 'min_child_weight': (20, 40),
        'min_child_samples': (15, 50),
    }

    bo = BayesianOptimization(
        partial(
            evaluator,
            train_df,
            target),
        params,
        random_state=512)
    bo.maximize(init_points=init_points, n_iter=n_iter)
    best_params = bo.max['params']
    best_params.update({'n_estimators': 10000, 'nthread': 4})
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])

    return best_params


def XGBoostBayesOptim(train_df, target, init_points=5, n_iter=15):
    params = {
        # 'base_score': (.5),
        # 'n_estimators': (1000),
        'colsample_bytree': (0.8, 1),
        'learning_rate': (.01, .14),
        'subsample': (0.8, 1),
        'max_depth': (4, 8),
        'gamma': (.1, .5),
        'reg_alpha': (.06, .5),
        'reg_lambda': (.03, .5),
        'min_child_weight': (4, 20),
        'min_child_samples': (15, 50),
        'scale_pos_weight': (1, 10),
        # 'seed': (42)
    }

    bo = BayesianOptimization(
        partial(
            xgb_evaluator,
            train_df,
            target),
        params,
        random_state=512)
    bo.maximize(init_points=init_points, n_iter=n_iter)
    best_params = bo.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    return best_params
