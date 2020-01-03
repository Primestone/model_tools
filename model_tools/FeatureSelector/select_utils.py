"""
feature selector utils

@author: Moon
"""

import numpy as np
from ..metrics import ks
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score


class GreedyFeatureSelection(object):

    def __init__(self, train_set, target_col, clf, good_features=[], verbose=True, test_set=None):
        self.train_set = train_set
        self.target_col = target_col
        self.test_set = test_set
        self.clf = clf
        self.columns = [x for x in self.train_set.columns if x != self.target_col]
        self._verbose = verbose
        self.good_features = good_features

    @staticmethod
    def evaluator(train, target_col, clf, n_folds=5, use_stratified=True, seed=42, verbose=False, test=None):
        variables = [x for x in train.columns if x != target_col]
        oof_predict = np.zeros(train.shape[0])
        if test is not None:
            oot_predict = np.zeros(test.shape[0])
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) if use_stratified else \
            KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for n_fold, (trn_idx, vld_idx) in enumerate(kf.split(train[variables], train[target_col])):
            x_trn, x_vld = train[variables].iloc[trn_idx, :], train[variables].iloc[vld_idx, :]
            y_trn, y_vld = train[target_col].values[trn_idx], train[target_col][vld_idx]
            clf.fit(x_trn, y_trn, eval_set=[(x_trn, y_trn), (x_vld, y_vld)], eval_metric='auc',
                    early_stopping_rounds=100, verbose=False)
            vld_predict = clf.predict_proba(x_vld)[:, 1]  # , num_iteration=clf.best_iteration_
            oof_predict[vld_idx] = vld_predict
            if test is not None:
                oot_predict += clf.predict_proba(test[variables])[:, 1] / kf.n_splits  # , num_iteration=clf.best_iteration_

        oof_score = round(roc_auc_score(train[target_col], oof_predict), 4)
        if verbose:
            print(f"- OOF Train Auc: [{oof_score}]")
            
        if test is not None:
            oot_score = round(roc_auc_score(test[target_col], oot_predict), 4)
            print(f"- OOT Auc: [{oot_score}]")
            if (oof_score - oot_score) > 0.035:
                score = 0
            else:
                score = oof_score #0.35*oof_score + 0.65*oot_score #
            return score
        else:
            return oof_score

    def selectionLoop(self):
        score_history = []
        if self.good_features is None:
            good_features = []
        else:
            good_features = self.good_features
        n_features = len(self.columns)

        columns_len = []
        for f in self.columns:
            columns_len.append(len(f))
        max_length = max(columns_len)

        loop = 1
        while (len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]) or \
                (len(score_history) >= 2 and (score_history[-1][0] > score_history[-2][0] or
                                              score_history[-1][0] > score_history[-3][0] or
                                              score_history[-1][0] > score_history[-4][0])):
            scores = []
            for i, feature in enumerate(self.columns):
                print(f"LOOP: {loop}, Process({i+1}/{n_features})")
                if feature not in good_features:
                    selected_features = good_features + [feature]
                    train = self.train_set[selected_features + [self.target_col]]
                    if self.test_set is not None:
                        test = self.test_set[selected_features + [self.target_col]]
                    else:
                        test = None

                    score = round(self.evaluator(train, self.target_col, self.clf, test=test), 4)
                    scores.append((score, feature))

                    if self._verbose:
                        print("- Join {0}: {1}".format(feature.ljust(max_length), np.mean(score)))

            good_features.append(sorted(scores)[-1][1])
            score_history.append(sorted(scores)[-1])
            if self._verbose:
               print("Current features : ", list(good_features))
            loop += 1
        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = list(good_features)
        if self._verbose:
            print("- Selected Features : ", good_features)

        return good_features

    def deletefeatureLoop(self):
        score_history = []
        full_features = self.columns
        n_features = len(full_features)

        loop = 1
        while (len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]) or \
                (len(score_history) >= 2 and (score_history[-1][0] > score_history[-2][0] or \
                                              score_history[-1][0] > score_history[-3][0] or \
                                              score_history[-1][0] > score_history[-4][0])):
            scores = []

            if len(score_history) < 1:
                features_to_use = full_features
                train = self.train_set[features_to_use + [self.target_col]]
                score = self.evaluator(train, self.target_col, self.clf)
                score_history.append((score, 'All'))

                if self._verbose:
                    print("- First Step: {} ({})".format(round(np.mean(score), 4), len(features_to_use)))

            else:
                for i, feature in enumerate(full_features):
                    print(f"LOOP: {loop}, Process({i+1}/{n_features})")
                    if feature not in self.good_features:
                        features_to_use = [x for x in full_features if x != feature]
                        train = self.train_set[features_to_use + [self.target_col]]
                        score = self.evaluator(train, self.target_col, self.clf)
                        scores.append((score, feature))

                        if self._verbose:
                            print("- Drop {0} : {1} ({2}/{3})".format(feature, round(np.mean(score), 6), i + 1,
                                                                      len(features_to_use)))

                full_features.remove(sorted(scores)[-1][1])
                score_history.append(sorted(scores)[-1])
                print(score_history)
                # if self._verbose:
                #    print("Current drop feature : ", sorted(scores)[-1][1])
            loop += 1
        # Remove last added feature
        full_features.append(score_history[-1][1])
        if self._verbose:
            print("Selected features : ", full_features)
        return full_features
