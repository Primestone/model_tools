# -*- coding: utf-8 -*-

"""
Stacking Framework

@author: Moon
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.cross_validation import StratifiedKFold, KFold
import pandas as pd
import os
import sys
import time
import logging

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import naive_bayes

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)


class Ensembler(object):
    def __init__(self, model_dict, params, num_folds=3, task_type='classification', optimize=roc_auc_score,
                 lower_is_better=False, save_path='./'):
        """
        Ensembler init function
        :param model_dict: model dictionary, see README for its format
        :param num_folds: the number of folds for ensembling
        :param task_type: classification or regression
        :param optimize: the function to optimize for, e.g. AUC, logloss, etc. Must have two arguments y_test and y_pred
        :param lower_is_better: is lower value of optimization function better or higher
        :param save_path: path to which model pickles will be dumped to along with generated predictions, or None
        """

        self.model_dict = model_dict
        self.levels = len(self.model_dict)
        self.num_folds = num_folds
        self.params = params
        self.task_type = task_type
        self.optimize = optimize
        self.lower_is_better = lower_is_better
        self.save_path = save_path

        self.training_data = None
        self.test_data = None
        self.y = None
        self.lbl_enc = None
        self.y_enc = None
        self.train_prediction_dict = None
        self.test_prediction_dict = None
        self.num_classes = None

    def turn_model(self, trn_X, trn_y, val_X, val_y, model_name):
        if model_name.startswith('Xgboost'):
            dtrn = xgb.DMatrix(trn_X, trn_y)
            dval = xgb.DMatrix(val_X, val_y)
            watchlist = [(dtrn, 'train'), (dval, 'valid')]
            xgb_params = self.params[model_name]
            model = xgb.train(xgb_params, dtrn, 10000, watchlist, early_stopping_rounds=200, verbose_eval=False)

        elif model_name.startswith('XGBClassifier'):
            xgb_params = self.params[model_name]
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(trn_X, trn_y, eval_set=[(trn_X, trn_y), (val_X, val_y)], eval_metric='auc',
                      early_stopping_rounds=150, verbose=False)

        elif model_name.startswith('Lightgbm'):
            valid_list = [lgb.Dataset(val_X, label=val_y, reference=lgb.Dataset(trn_X, label=trn_y))]
            lgb_params = self.params[model_name]
            model = lgb.train(lgb_params, lgb.Dataset(trn_X, label=trn_y), num_boost_round=10000, valid_sets=valid_list,
                              early_stopping_rounds=200, verbose_eval=False)

        elif model_name.startswith('LGBMClassifier'):
            lgb_params = self.params[model_name]
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(trn_X, trn_y, eval_set=[(trn_X, trn_y), (val_X, val_y)], eval_metric="auc",
                      early_stopping_rounds=150, verbose=False)

        elif model_name.startswith('Catboost'):
            cat_params = self.params[model_name]
            model = CatBoostClassifier(**cat_params)
            model.fit(trn_X, trn_y, eval_set=[val_X, val_y], use_best_model=True, verbose=False)

        elif model_name.startswith('NN'):
            pass

        elif model_name.startswith('MLP'):
            mlp_params = self.params[model_name]
            model = MLPClassifier(**mlp_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('RandomForest'):
            rf_params = self.params[model_name]
            model = RandomForestClassifier(**rf_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('GBDT'):
            gbm_params = self.params[model_name]
            model = GradientBoostingClassifier(**gbm_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('Adaboost'):
            ada_params = self.params[model_name]
            model = AdaBoostClassifier(**ada_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('ET'):
            et_params = self.params[model_name]
            model = ExtraTreesClassifier(**et_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('Ovr'):
            ovr_params = self.params[model_name]
            model = OneVsRestClassifier(**ovr_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('Logistic'):
            lr_params = self.params[model_name]
            model = LogisticRegression(**lr_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('Lasso'):
            lasso_params = self.params[model_name]
            model = Lasso(**lasso_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('Ridge'):
            ridge_params = self.params[model_name]
            model = RidgeClassifier(**ridge_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('Knn'):
            knn_params = self.params[model_name]
            model = KNeighborsClassifier(**knn_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('SVM'):
            svm_params = self.params[model_name]
            model = SVC(**svm_params)
            model.fit(trn_X, trn_y)

        elif model_name.startswith('Bayes'):
            bayes_params = self.params[model_name]
            model = naive_bayes(**bayes_params)
            model.fit(trn_X, trn_y)

        return model

    def stacking(self, training_data, y, lentrain, test_data, lentest):
        """
        :param training_data: training data in tabular format
        :param y: binary, multi-class or regression
        :param lentrain: train rows(train.shape[0])
        :param test_data: test dataset
        :param lentest: test rows(test.shape[0])
        :return: chain of models to be used in prediction
        """

        self.training_data = training_data
        self.y = y

        self.test_data = test_data

        if self.task_type == 'classification':
            self.num_classes = len(np.unique(self.y))
            logger.info("Found %d classes", self.num_classes)
            self.lbl_enc = LabelEncoder()
            self.y_enc = self.lbl_enc.fit_transform(self.y)
            kf = StratifiedKFold(y=self.y_enc, n_folds=self.num_folds, shuffle=True, random_state=2017)
            train_prediction_shape = (lentrain, self.num_classes)
            test_prediction_shape = (lentest, self.num_classes)

        else:
            self.num_classes = -1
            self.y_enc = self.y
            kf = KFold(n=len(self, y), n_folds=self.num_folds, shuffle=True, random_state=2017)
            train_prediction_shape = (lentrain, 1)
            test_prediction_shape = (lentest, 1)

        self.train_prediction_dict = {}
        self.test_prediction_dict = {}

        for level in range(self.levels):
            self.train_prediction_dict[level] = np.zeros((train_prediction_shape[0],
                                                          train_prediction_shape[1] * len(self.model_dict[level])))

            self.test_prediction_dict[level] = np.zeros((test_prediction_shape[0],
                                                         test_prediction_shape[1] * len(self.model_dict[level])))

        for level in range(self.levels):

            if level == 0:
                temp_train = self.training_data
                temp_test = self.test_data
            else:
                temp_train = self.train_prediction_dict[level - 1]
                temp_test = self.test_prediction_dict[level - 1]

            columns = []
            for model_num, model_name in enumerate(self.model_dict[level]):
                columns.extend(model_name + '_' + str(i) for i in range(self.num_classes))
                validation_scores = []
                foldnum = 1

                temp_test_predictions = np.zeros((test_prediction_shape[0], self.num_classes))

                for train_index, valid_index in kf:
                    logger.info("Training Level %d Fold # %d. Model # %s", level, foldnum, model_name)

                    if level != 0:
                        l_training_data = temp_train[train_index]
                        l_validation_data = temp_train[valid_index]
                        model = self.turn_model(temp_train[train_index], self.y_enc[train_index],
                                                temp_train[valid_index], self.y_enc[valid_index], model_name)

                    else:
                        l0_training_data = temp_train[0][0]
                        if type(l0_training_data) == list:
                            l_training_data = [x[train_index] for x in l0_training_data]
                            l_validation_data = [x[valid_index] for x in l0_training_data]
                        else:
                            l_training_data = l0_training_data[train_index]
                            l_validation_data = l0_training_data[valid_index]
                        model = self.turn_model(l_training_data, self.y_enc[train_index], l_validation_data,
                                                self.y_enc[valid_index], model_name)

                    logger.info("Predicting Level %d. Fold # %d. Model # %s", level, foldnum, model_name)

                    if self.task_type == 'classification':
                        if model_name == 'Xgboost':
                            tmp_preds = model.predict(xgb.DMatrix(l_validation_data),
                                                      ntree_limit=model.best_ntree_limit)
                            temp = np.zeros((len(tmp_preds), self.num_classes))
                            temp[:, 1] = tmp_preds
                            temp[:, 0] = 1 - tmp_preds
                            temp_train_predictions = temp
                        elif model_name == 'Lightgbm':
                            tmp_preds = model.predict(l_validation_data, num_iteration=model.best_iteration + 1)
                            temp = np.zeros((len(tmp_preds), self.num_classes))
                            temp[:, 1] = tmp_preds
                            temp[:, 0] = 1 - tmp_preds
                            temp_train_predictions = temp

                        else:
                            temp_train_predictions = model.predict_proba(l_validation_data)

                        self.train_prediction_dict[level][valid_index,
                        (model_num * self.num_classes):(model_num * self.num_classes) +
                                                       self.num_classes] = temp_train_predictions
                        if level == 0:
                            if model_name == 'Xgboost':
                                xgb_preds = model.predict(xgb.DMatrix(temp_test[0][0]),
                                                          ntree_limit=model.best_ntree_limit)
                                temp = np.zeros((len(xgb_preds), self.num_classes))
                                temp[:, 1] = xgb_preds
                                temp[:, 0] = 1 - xgb_preds
                                temp_test_predictions += temp / self.num_folds
                            elif model_name == 'Lightgbm':
                                lgb_preds = model.predict(temp_test[0][0], num_iteration=model.best_iteration + 1)
                                temp = np.zeros((len(lgb_preds), self.num_classes))
                                temp[:, 1] = lgb_preds
                                temp[:, 0] = 1 - lgb_preds
                                temp_test_predictions += temp / self.num_folds
                            else:
                                temp_test_predictions += model.predict_proba(temp_test[0][0]) / self.num_folds

                        else:
                            if model_name == 'Xgboost':
                                xgb_preds = model.predict(xgb.DMatrix(temp_test), ntree_limit=model.best_ntree_limit)
                                temp = np.zeros((len(xgb_preds), self.num_classes))
                                temp[:, 1] = xgb_preds
                                temp[:, 0] = 1 - xgb_preds
                                temp_test_predictions += temp / self.num_folds
                            elif model_name == 'Lightgbm':
                                lgb_preds = model.predict(temp_test, num_iteration=model.best_iteration + 1)
                                temp = np.zeros((len(lgb_preds), self.num_classes))
                                temp[:, 1] = lgb_preds
                                temp[:, 0] = 1 - lgb_preds
                                temp_test_predictions += temp / self.num_folds
                            else:
                                temp_test_predictions += model.predict_proba(temp_test) / self.num_folds

                    else:
                        temp_train_predictions = model.predict(l_validation_data)
                        self.train_prediction_dict[level][valid_index, model_num] = temp_train_predictions
                        if level == 0:
                            temp_test_predictions += model.predict(temp_test[0][0]) / self.num_folds
                        else:
                            temp_test_predictions += model.predict(temp_test) / self.num_folds

                    if self.num_classes >= 3:
                        validation_score = self.optimize(self.y_enc[valid_index], temp_train_predictions)
                    else:
                        validation_score = self.optimize(self.y_enc[valid_index], temp_train_predictions[:, 1])
                    validation_scores.append(validation_score)
                    logger.info("Level %d. Fold # %d. Model # %s. Validation Score = %f", level, foldnum, model_num,
                                validation_score)
                    foldnum += 1

                self.test_prediction_dict[level][:, (model_num * self.num_classes): (model_num * self.num_classes) +
                                                                            self.num_classes] = temp_test_predictions
                avg_score = np.mean(validation_scores)
                std_score = np.std(validation_scores)
                logger.info("Level %d. Model # %d. Mean Score = %f. Std Dev = %f", level, model_num,
                            avg_score, std_score)

            # logger.info("Saving predictions for level # %d", level)
            t1 = str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
            train_predictions_df = pd.DataFrame(data=self.train_prediction_dict[level], columns=columns)
            train_predictions_df.to_csv(
                os.path.join(self.save_path, "train_predictions_level_" + str(level) + '_' + t1 + ".csv"),
                index=False)

            test_predictions_df = pd.DataFrame(data=self.test_prediction_dict[level], columns=columns)
            test_predictions_df.to_csv(
                os.path.join(self.save_path, "test_predictions_level_" + str(level) + '_' + t1 + ".csv"),
                index=False)

        return self.train_prediction_dict, self.test_prediction_dict
