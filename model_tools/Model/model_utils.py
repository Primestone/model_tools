# -*- coding: utf-8 -*-

"""
KFolds Classifier Base on Scikit-Learn wrapper. (LightGBM, XGBoost...)
"""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from ..metrics import ks


def KfoldClassifier(
        train,
        target_col,
        test,
        clf,
        n_folds=5,
        seed=512,
        use_stratified=True,
        verbose=False,
        eval_metric='auc',
        save_model=None):
    variables = [x for x in train.columns if x != target_col]
    oof_predict = np.zeros(train.shape[0])
    oot_predict = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    model_file = {}
    model_list = []
    score_list = []
    kf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed) if use_stratified else KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed)

    for n_fold, (trn_idx, vld_idx) in enumerate(
            kf.split(train[variables], train[target_col])):
        x_trn, x_vld = train[variables].iloc[trn_idx,
                                             :], train[variables].iloc[vld_idx, :]
        y_trn, y_vld = train[target_col].values[trn_idx], train[target_col][vld_idx]
        clf.fit(
            x_trn,
            y_trn,
            eval_set=[
                (x_trn,
                 y_trn),
                (x_vld,
                 y_vld)],
            eval_metric=eval_metric,
            early_stopping_rounds=100,
            verbose=False)
        vld_predict = clf.predict_proba(x_vld)[
            :, 1]  # , num_iteration=clf.best_iteration_
        oof_predict[vld_idx] = vld_predict
        vld_score = roc_auc_score(y_vld, vld_predict)
        vld_ks = ks(y_vld, vld_predict)
        # , num_iteration=clf.best_iteration_
        oot_predict += clf.predict_proba(test[variables])[:, 1] / kf.n_splits

        fold_model = clf.booster_.dump_model()
        model_file[f'fold{n_fold+1}_tree'] = fold_model

        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = variables
        fold_importance_df['importance'] = clf.feature_importances_
        fold_importance_df['fold'] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
        if verbose:
            print(
                '- Fold %d AUC : %.4f, KS : %.4f' %
                (n_fold + 1, vld_score, vld_ks))
        model_list.append(clf)
        score_list.append(vld_score)

    oof_score = round(roc_auc_score(train[target_col], oof_predict), 4)
    oof_ks = round(ks(train[target_col], oof_predict), 4)
    score_std = round(np.std(score_list), 4)
    if verbose:
        print(f"- OOF Train Auc: [{oof_score}], STD: [{score_std}]")
        print(f"- OOF Train KS : [{oof_ks}]")
    feature_importance = feature_importance_df[["feature", "importance"]].groupby(
        "feature").mean().sort_values(by="importance", ascending=False).reset_index()
    feature_importance.columns = ['variables', 'importance']

    model_file['ecdf'] = list(oof_predict)
    if save_model is not None:
        with open(f'{save_model}.json', 'w') as f:
            json.dump(model_file, f)

    return model_list, oot_predict, oof_predict, score_list, feature_importance


def GreedyThresholdSelector(
        data,
        target_col,
        test,
        clf,
        stats,
        trend_corr_range,
        n_folds=5,
        select_min=50,
        select_limit=100,
        seed_list=[88, 1001],
    eval_metric='auc',
        verbose=False):
    result = pd.DataFrame()
    for seed in seed_list:
        for tc in trend_corr_range:
            print('- Trend Correlation: {0}'.format(round(tc, 3)))
            sel_cols = stats[stats['Trend_correlation']
                             >= tc]['Feature'].tolist()
            assert len(sel_cols) >= 1, 'select features must more than four. '

            _, _, _, _, feature_imp = KfoldClassifier(
                data[sel_cols + [target_col]], target_col, test, clf, n_folds=n_folds, seed=seed, eval_metric=eval_metric)
            max_sels = feature_imp.shape[0] + 1
            max_sels = min(max_sels, select_limit)
            for n in range(select_min, max_sels):
                print(
                    '- Seed: {0}. Trend Correlation: {1}. Num Features: {2}/{3}'.format(
                        seed,
                        round(
                            tc,
                            3),
                        n,
                        max_sels -
                        1))
                top_cols = feature_imp['variables'].tolist()[: n]
                _, oot_predict, oof_predict, score_list, _ = KfoldClassifier(
                    data[top_cols + [target_col]], target_col, test, clf, n_folds=n_folds, seed=seed)

                oof_train_auc = round(
                    roc_auc_score(
                        data[target_col],
                        oof_predict),
                    4)
                oof_train_ks = round(ks(data[target_col], oof_predict), 4)
                cv_std = round(np.std(score_list), 4)
                test_auc = round(
                    roc_auc_score(
                        test[target_col],
                        oot_predict),
                    4)
                test_ks = round(ks(test[target_col], oot_predict), 4)
                if verbose:
                    print(
                        f" - OOF Train Auc: [{oof_train_auc}], STD: [{cv_std}]")
                    print(f" - OOF Train KS : [{oof_train_ks}]")
                params = clf.get_params()
                res = pd.DataFrame(
                    data=[
                        seed,
                        n_folds,
                        tc,
                        top_cols,
                        len(top_cols),
                        oof_train_auc,
                        cv_std,
                        oof_train_ks,
                        test_auc,
                        test_ks,
                        params]).T
                res.columns = [
                    'seed',
                    'kfolds',
                    'trend_correlation',
                    'sub_columns',
                    'num_variables',
                    'train_auc',
                    'auc_std',
                    'train_ks',
                    'test_auc',
                    'test_ks',
                    'params']
                result = result.append(res)
    return result.reset_index(drop=True)
