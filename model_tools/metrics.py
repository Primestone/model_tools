# -*- coding: utf-8 -*-

"""
Common Evaluation Metrics

@author: Moon
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold


def rmse(y_ture, y_pred):
    return mean_squared_error(y_ture, y_pred) ** 0.5


def gini(y_ture, y_pred):
    assert (len(y_ture) == len(y_pred))

    all = np.asarray(np.c_[y_ture, y_pred, np.arange(len(y_ture))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    total_loss = all[:, 0].sum()
    sum = all[:, 0].cumsum().sum() / total_loss

    sum -= (len(y_ture) + 1) / 2.
    return sum / len(y_ture)


def gini_normalized(y_ture, y_pred):
    return gini(y_ture, y_pred) / gini(y_ture, y_ture)


def logloss(y_ture, y_pred):
    y_pred = max(min(y_pred, 1. - 10e-15), 10e-15)
    return -np.log(y_pred) if y_ture == 1. else -np.log(1. - y_pred)


def rmspe(y_ture, y_pred):
    w = np.zeros(y_ture.shape, dtype=float)
    ind = y_ture != 0
    w[ind] = 1. / (y_ture[ind] ** 2)
    rmspe = np.sqrt(np.mean(w * (y_ture - y_pred) ** 2))
    return rmspe


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def ndgc_k(y_ture, y_pred, k=5):
    top = []
    for i in range(y_pred.shape[0]):
        top.append(np.argsort(y_pred[i])[::-1][:k])
    mat = np.reshape(np.repeat(y_ture, np.shape(top)[1]) == np.array(top).ravel(), np.array(top).shape).astype(int)
    return np.mean(np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))


def ndgc5(y_ture, y_pred):
    top = []
    for i in range(y_pred.shape[0]):
        top.append(np.argsort(y_pred[i])[::-1][:5])
    mat = np.reshape(np.repeat(y_ture, np.shape(top)[1]) == np.array(top).ravel(), np.array(top).shape).astype(int)
    return np.mean(np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))


def ndgc10(y_ture, y_pred):
    top = []
    for i in range(y_pred.shape[0]):
        top.append(np.argsort(y_pred[i])[::-1][:10])
    mat = np.reshape(np.repeat(y_ture, np.shape(top)[1]) == np.array(top).ravel(), np.array(top).shape).astype(int)
    return np.mean(np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))


def ap_at_k(y_ture, y_pred, k=5):
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_pred):
        if p in y_ture and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not y_ture:
        return 0.0

    return score / min(len(y_ture), k)


def map_at_k(y_ture, y_pred, k=5):
    return np.mean([ap_at_k(a, p, k) for a, p in zip(y_ture, y_pred)])


def map5(y_ture, y_pred):
    return map_at_k(y_ture, y_pred, 5)


def map10(y_ture, y_pred):
    return map_at_k(y_ture, y_pred, 10)


def ks(y_true, y_pred):
    fpr, tpr, thre = roc_curve(y_true, y_pred, pos_label=1)
    return abs(fpr-tpr).max()


def lgb_ks(preds, train_data):
    true = train_data.get_label()
    fpr, tpr, thre = roc_curve(true, preds, pos_label=1)
    return 'ks', abs(fpr-tpr).max(), True


def xgb_ks(preds, dtrain):
    label = dtrain.get_label()
    fpr, tpr, thre = roc_curve(label, preds, pos_label=1)
    return 'ks', abs(fpr-tpr).max()


def lift(y_true, y_pred, thread=0.5):
    preds_label = [1 if x > thread else 0 for x in y_pred]
    cm = confusion_matrix(y_true, preds_label)
    pv = cm[1][1]/(cm[0][1]+cm[1][1])
    k = (cm[1][0]+cm[1][1])/(sum(sum(cm)))
    lift = pv/k
    return lift


def get_cv_score(X, y, clf, params, n_folds=5, metric='auc',
                 stratified=True, shuffle=True, seed=512):
    """
    N folds cross-validation score list based on classifier

    Paramters
    ---------
    clf: model classifier
    params: paramters of the classifier u used
    n_folds: cross validation flods
    metric: evalution function

    Return
    ------
    score list: score list based on metric you used
    mean score: mean score of n folds evaluation
    """

    metrics_dict = {'auc': roc_auc_score,
                    'ks': ks,
                    'f1': f1_score,
                    'gini': gini_normalized,
                    'rmse': rmse}

    assert metric in metrics_dict.keys(), "metrics dict not contain {}, add it if you need".format(metric)

    # TODO
    if not (hasattr(clf, 'fit') and hasattr(clf, 'predict_prob')):
        raise ValueError("The classifier must have fit and predict attribute. ")

    score_list = []
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    else:
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    for trn_ind, val_ind in kf.split(X, y):
        if isinstance(X, np.ndarray):
            X_trn, X_val = X[trn_ind], X[val_ind]
            y_trn, y_val = y[trn_ind], y[val_ind]
        elif isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
            X_trn, X_val = X.iloc[trn_ind, :], X.iloc[val_ind, :]
            y_trn, y_val = y.values[trn_ind], y[val_ind]
        model = clf.fit(X_trn, y_trn)
        y_pred = model.predict_prob(X_val)[:, 1]
        score = metrics_dict[metric](y_val, y_pred)
        score_list.append(score)
    score_avg = np.mean(score_list)
    score_std = np.std(score_list)
    print("{0} folds. cross validation get {1}: {2}±{3}.".format(n_folds, metric, round(score_avg, 4),
                                                                 round(score_std, 4)))
    return score_list, score_avg
