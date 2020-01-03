# -*- coding: utf-8 -*-
"""
Auto XGBoost and Auto Scorecard
@author: tangyangyang
"""
import pickle
import cloudpickle
import varclushi
import numpy as np
import pandas as pd
import scorecardpy as sc
from abc import abstractmethod
from .ScoreCard import model_helper, modeler
from sklearn.pipeline import Pipeline
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from .data import DataHelper
from .Preprocessing.stabler import get_trend_stats
from .metrics import roc_auc_score, ks
from .Model.model_utils import GreedyThresholdSelector
from .Model.params_tune import XGBoostBayesOptim
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


def is_numpy(x):
    return isinstance(x, np.ndarray)


def is_pandas(x):
    return isinstance(x, pd.DataFrame)


class CategoryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, first_category=1, copy=True):
        self.categorical_features = categorical_features
        self.first_category = first_category
        self.copy = copy
        self.encoders = {}
        self.ive = None

    def fit(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")

        x = X[self.categorical_features].fillna("")
        self.encoders = {}
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in self.categorical_features:
            try:
                enc = LabelEncoder().fit(x.loc[:, i])
                self.encoders.update({i: enc})
            except BaseException:
                ind_v = x.loc[:, i].drop_duplicates().fillna(
                    "").reset_index(drop=True).to_dict()
                v_ind = {v: k for k, v in ind_v.items()}
                self.encoders.update({i: v_ind})

        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input X must a dataframe!")

        inner_categorical = [
            x for x in X.columns if x in self.categorical_features]
        x = X[inner_categorical].fillna("")
        if self.copy:
            x = X[inner_categorical].fillna("").copy()

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in inner_categorical:
            enc = self.encoders[i]
            if hasattr(enc, 'transform'):
                x.loc[:, i] = enc.transform(x.loc[:, i]) + self.first_category
            else:
                x.loc[:, i] = x.loc[:, i].fillna("").map(enc)
        X[inner_categorical] = x.values

        return X


class CountEncoder(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            categorical_features,
            min_count=0,
            nan_value=-1,
            copy=True):
        self.categorical_features = categorical_features
        self.min_count = min_count
        self.nan_value = nan_value
        self.copy = copy
        self.counts = {}

    def fit(self, X, y=0):
        self.counts = {}
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        x = X[self.categorical_features].fillna("")
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in self.categorical_features:
            cnt = x.loc[:, i].value_counts().to_dict()
            if self.min_count > 0:
                cnt = dict((k, self.nan_value if v < self.min_count else v)
                           for k, v in cnt.items())
            self.counts.update({i: cnt})
        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):
        if not is_pandas(X):
            raise TypeError("Input x must a dataframe!")

        inner_categorical = [
            x for x in X.columns if x in self.categorical_features]
        x = X[inner_categorical].fillna("")

        if self.copy:
            x = X[inner_categorical].fillna("").copy()

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        for i in inner_categorical:
            cnt = self.counts[i]
            x.loc[:, i] = x.loc[:, i].map(cnt)
        x = x.add_prefix('cnt_enc_')
        output = pd.concat([X, x], axis=1)
        return output


class AutoXGBoost(object):

    def __init__(
            self,
            train,
            test,
            target,
            key,
            unuse_variables,
            date_cols,
            missing_rate_threshold,
            select_min,
            max_features_num,
            trend_correlation_list,
            output_file,
            project_name,
            bin_method='tree',
            bin_num=9):

        self.train = train
        self.test = test
        self.target = target
        self.key = key
        self.unuse_variables = unuse_variables
        self.date_cols = date_cols
        self.missing_rate_threshold = missing_rate_threshold
        self.select_min = select_min
        self.max_features_num = max_features_num
        self.bin_method = bin_method
        self.bin_num = bin_num
        self.trend_correlation_list = trend_correlation_list
        self.output_file = output_file
        self.project_name = project_name
        assert self.key not in self.unuse_variables, "key must not in unuse_variables. "

    @abstractmethod
    def _data_preprocess(self, **kwargs):
        return self

    def _data_combine(self):
        self.train = self.train[[
            x for x in self.train.columns if x not in self.unuse_variables]]
        self.test = self.test[[
            x for x in self.test.columns if x not in self.unuse_variables]]
        self.datahper = DataHelper(target=self.target,
                                   train_path=None,
                                   test_path=None,
                                   trainfile=self.train,
                                   testfile=self.test,
                                   date_cols=self.date_cols)
        data = self.datahper.combine()
        dtypes = data.dtypes
        object_features = self.datahper.object_features
        self.categorical_variables = dtypes[dtypes == 'object'].index.tolist() if len(
            object_features) == 0 else object_features
        self.continues_features = self.datahper.continues_features
        self.bool_variables = dtypes[dtypes == 'bool'].index.tolist()
        for col in self.bool_variables:
            data[col] = data[col].apply(lambda x: 1 if x else 0)
        self.data = data
        return self

    def _data_report(self):
        self.data_report = model_helper.data_report(self.train)
        self.data_report.to_excel(
            self.output_file +
            "{0}_Data_Report.xlsx".format(self.project_name),
            index=False)
        print("DEV SAMPLE SIZE: %d, BAD RATIO: %.4f" %
              (self.train.shape[0], self.train[self.target].mean()))
        print("OOT SAMPLE SIZE: %d, BAD RATIO: %.4f" %
              (self.test.shape[0], self.test[self.target].mean()))
        return self

    def _feature_engineer(self):
        pipe = Pipeline([
            ('CatgoryEncoder', CategoryEncoder(self.categorical_variables)),
            ('CountEncoder', CountEncoder(self.categorical_variables)),
        ])
        self.data = pipe.fit_transform(self.data)
        self.train, self.test = self.datahper.split(self.data)
        self.data_report = model_helper.data_report(self.train)

        variable_filter = self.data_report[self.data_report['unique'] > 1]
        variable_filter = variable_filter[~(
            (variable_filter['dtype'] == 'object') & (variable_filter['unique'] >= 20))]
        variable_filter = variable_filter[variable_filter['missing_rate']
                                          <= self.missing_rate_threshold]
        variable_filter = variable_filter[variable_filter['dtype']
                                          != 'datetime64[ns]']
        use_cols = list(variable_filter['column'])
        use_cols = [x for x in use_cols if x not in [self.key]]

        self.train[use_cols] = self.train[use_cols].fillna(-999)
        self.test[use_cols] = self.test[use_cols].fillna(-999)
        self.use_cols = use_cols

        cloudpickle.dump(
            pipe,
            open(
                self.output_file +
                "{0}_pipeline.pkl".format(self.project_name),
                "wb"))

    def _feature_select(self):
        self._data_preprocess()
        self._data_combine()
        self._data_report()
        self._feature_engineer()

        stats = get_trend_stats(
            data=self.train[self.use_cols],
            target_col=self.target,
            bins=self.bin_num,
            data_test=self.test[self.use_cols],
            method=self.bin_method)
        stats.sort_values('Trend_correlation', ascending=False, inplace=True)
        stats.to_excel(
            self.output_file +
            "{0}_Trend_Correlation.xlsx".format(self.project_name),
            index=False)
        self.stats = stats

        gbm_model = LGBMClassifier(boosting_type='gbdt', num_leaves=2 ** 5, max_depth=5,
                                   learning_rate=0.1, n_estimators=10000,  # class_weight=20,
                                   min_child_samples=20,
                                   subsample=0.95, colsample_bytree=0.95,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   sample_weight=None, seed=1001  # init_score=0.5
                                   )

        result = GreedyThresholdSelector(
            self.train[self.use_cols],
            self.target,
            self.test[self.use_cols],
            gbm_model,
            stats,
            self.trend_correlation_list,
            5,
            self.select_min,
            self.max_features_num,
            [1001])
        self.gs_result = result

        result.to_excel(
            self.output_file +
            "{0}_GS_Result.xlsx".format(self.project_name),
            index=False)
        init_variables = result.sort_values('test_auc', ascending=False).head(1)[
            'sub_columns'].values[0]
        self.init_variables = init_variables

    def xgboost_model(
            self,
            proba_index=0,
            sel_cols=None,
            xgboost_params=None,
            save=True):
        assert hasattr(
            self, 'init_variables') or sel_cols is not None, "Before run xgboost model, must select variables."
        if (not hasattr(self, 'init_variables')) and sel_cols is not None and not hasattr(self, 'sel_cols'):
            self._data_preprocess()
            self._data_combine()
            self._data_report()
            self._feature_engineer()

        if xgboost_params is None:
            model = XGBClassifier(
                base_score=0.5,
                colsample_bylevel=1,
                colsample_bytree=0.843,
                gamma=0.1,
                learning_rate=0.05,
                max_delta_step=0,
                max_depth=4,
                min_child_weight=5,
                missing=None,
                n_estimators=10000,
                nthread=-1,
                n_jobs=-1,
                objective='binary:logistic',
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=1,
                seed=42,
                silent=True,
                subsample=0.9)
        else:
            model = XGBClassifier(**xgboost_params)

        if sel_cols is None and hasattr(self, 'init_variables'):
            sel_cols = self.init_variables

        self.sel_cols = sel_cols

        params = model.get_xgb_params()
        xgb_train = xgb.DMatrix(
            self.train[sel_cols], label=self.train[self.target])
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
        print('Best round num: ', num_round_best)
        # train
        params['n_estimators'] = num_round_best

        model = XGBClassifier(**params)
        model.fit(self.train[sel_cols], self.train[self.target])
        self.model = model
        dev_xgb_predict = model.predict_proba(
            self.train[sel_cols])[:, proba_index]
        oot_xgb_predict = model.predict_proba(
            self.test[sel_cols])[:, proba_index]

        self.train['proba'] = dev_xgb_predict
        self.test['proba'] = oot_xgb_predict

        auc_mean = round(max(cv_result['test-auc-mean']), 4)
        auc_std = round(max(cv_result['test-auc-std']), 4)
        print(f'- 5Folds AUC: {auc_mean}, STD: {auc_std}')
        print('- Test AUC: %.4f' %
              roc_auc_score(self.test[self.target], 1 - self.test['proba']))
        print('- Test KS : %.4f' %
              ks(self.test[self.target], 1 - self.test['proba']))

        importance_df = pd.DataFrame()
        importance_df['feature'] = sel_cols
        importance_df['importance'] = model.feature_importances_
        if save:
            pickle.dump(
                model, open(
                    self.output_file +
                    '{0}_model.pkl'.format(self.project_name), 'wb'))
            pickle.dump(
                sel_cols, open(
                    self.output_file +
                    "{0}_input_variables.pkl".format(
                        self.project_name), 'wb'))

            importance_df.to_excel(
                self.output_file +
                "{0}_Feature_Importance.xlsx".format(self.project_name),
                index=False)

            group_analysis = model_helper.model_group_monitor(
                self.test, self.target, 'proba')
            group_analysis.to_excel(
                self.output_file +
                "{0}_Group_Analysis.xlsx".format(self.project_name),
                index=False)

            self.test.to_excel(
                self.output_file +
                "{0}_Test_Sample.xlsx".format(self.project_name),
                index=False)

    def optim_params(self, init_points=5, n_iter=15):
        best_params = XGBoostBayesOptim(
            self.train[self.sel_cols + [self.target]], self.target,
            init_points=init_points, n_iter=n_iter)
        self.best_params = best_params
        return self


class AutoScoreCard(object):

    def __init__(self, train,
                 test,
                 target,
                 key,
                 unuse_variables,
                 date_cols,
                 missing_rate_threshold,
                 bin_num_limit,
                 bin_method,
                 output_file,
                 project_name):
        self.train = train
        self.test = test
        self.target = target
        self.key = key
        self.unuse_variables = unuse_variables
        self.date_cols = date_cols
        self.missing_rate_threshold = missing_rate_threshold
        self.bin_num_limit = bin_num_limit
        self.bin_method = bin_method
        self.output_file = output_file
        self.project_name = project_name
        assert self.key not in self.unuse_variables, "key must not in unuse_variables. "

    def _data_preprocess(self):
        return self

    def _data_combine(self):
        self.train = self.train[[
            x for x in self.train.columns if x not in self.unuse_variables]]
        self.test = self.test[[
            x for x in self.test.columns if x not in self.unuse_variables]]
        self.datahper = DataHelper(target=self.target,
                                   train_path=None,
                                   test_path=None,
                                   trainfile=self.train,
                                   testfile=self.test,
                                   date_cols=self.date_cols)
        data = self.datahper.combine()
        dtypes = data.dtypes
        object_features = self.datahper.object_features
        self.categorical_variables = dtypes[dtypes == 'object'].index.tolist() if len(
            object_features) == 0 else object_features
        self.continues_features = self.datahper.continues_features
        self.bool_variables = dtypes[dtypes == 'bool'].index.tolist()
        for col in self.bool_variables:
            data[col] = data[col].apply(lambda x: 1 if x else 0)
        self.data = data
        return self

    def _data_report(self):
        self.data_report = model_helper.data_report(self.train)
        self.data_report.to_excel(
            self.output_file +
            "{0}_Data_Report.xlsx".format(self.project_name),
            index=False)
        print("DEV SAMPLE SIZE: %d, BAD RATIO: %.4f" %
              (self.train.shape[0], self.train[self.target].mean()))
        print("OOT SAMPLE SIZE: %d, BAD RATIO: %.4f" %
              (self.test.shape[0], self.test[self.target].mean()))
        return self

    def _variables_filter(self):
        self.train, self.test = self.datahper.split(self.data)
        self.data_report = model_helper.data_report(self.train)

        variable_filter = self.data_report[self.data_report['unique'] > 1]
        variable_filter = variable_filter[~(
                (variable_filter['dtype'] == 'object') & (variable_filter['unique'] >= 30))]
        variable_filter = variable_filter[variable_filter['missing_rate']
                                          <= self.missing_rate_threshold]
        variable_filter = variable_filter[variable_filter['dtype']
                                          != 'datetime64[ns]']
        use_cols = list(variable_filter['column'])
        use_cols = [x for x in use_cols if x not in [self.key]]
        self.use_cols = use_cols
        return self

    def binWoe(self):
        bins = sc.woebin(self.train[self.use_cols], y=self.target, bin_num_limit=self.bin_num_limit, method=self.bin_method)
        pickle.dump(bins, open(self.output_file+'{0}_{1}_bins.pkl'.format(self.project_name, self.bin_method), 'wb'))
        bins_df = pd.DataFrame()
        for k, v in bins.items():
            bins_df = pd.concat([bins_df, v])

        bins_df.sort_values(['total_iv', 'breaks'], ascending=[
            False, True], inplace=True)
        bins_df.to_excel(self.output_file+'{0}_{1}_bins_df.xlsx'.format(self.project_name, self.bin_method), index=False)
        self.dev_woe = sc.woebin_ply(self.train[self.use_cols], bins, no_cores=4)
        self.oot_woe = sc.woebin_ply(self.test[self.use_cols], bins, no_cores=4)
        self.bins = bins
        self.bins_df = bins_df
        return self

    def varclus_proc(self):
        self._data_combine()
        self._data_report()
        self._variables_filter()
        self.binWoe()
        sel_cols = self.bins_df[self.bins_df['total_iv'] >=
                           0.02]['variable'].drop_duplicates().tolist()
        dtypes = self.train[sel_cols].dtypes
        numeric_cols = dtypes[dtypes != 'object'].index.tolist()

        varclus_input = [x + '_woe' for x in numeric_cols if x != self.target]
        varclus_proc = varclushi.VarClusHi(self.dev_woe[varclus_input],
                                           maxeigval2=1, maxclus=None)
        varclus_proc.varclus()
        self.varclus_info = varclus_proc.info
        self.varclus_result = varclus_proc.rsquare
        self.varclus_result.to_excel(self.output_file+'{0}_varclus_result.xlsx'.format(self.project_name), index=False)
        varclus_iv = self.varclus_result.merge(
            self.bins_df[['Variable', 'total_iv']].drop_duplicates(), on='Variable', how='left')
        varclus_iv.sort_values(by=['Cluster', 'total_iv'],
                               ascending=[True, False], inplace=True)
        varclus_iv['rank'] = varclus_iv.groupby('Cluster')['total_iv'].rank(ascending=False)
        varclus_iv.to_excel(self.output_file+'{0}_varclus_iv.xlsx'.format(self.project_name), index=False)
        sw_df = varclus_iv[varclus_iv['rank']<=2].sort_values('total_iv', ascending=False)
        self.sw_input = sw_df['Variable'].tolist()[:70]
        return self

    def stepwise(self, sw_input=None):
        if sw_input is not None:
            sw_input = sw_input
        else:
            sw_input = self.sw_input
        sw = modeler.StepwiseModel(self.dev_woe[self.sw_input], self.dev_woe[self.target],
                                   method='stepwise')
        sw_result = sw.stepwise()
        self.input_cols = [x for x in sw_result.keys() if x != 'const']
        return self

    def logit_fit(self):
        lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
        lr.fit(self.dev_woe[self.input_cols], self.dev_woe[self.target])
        pickle.dump(lr, open(self.output_file+"{0}_LRModel.pkl".format(self.project_name), 'wb'))
        dev_pred = lr.predict_proba(self.dev_woe[self.input_cols])[:, 1]
        oot_pred = lr.predict_proba(self.oot_woe[self.input_cols])[:, 1]
        print("="*6+" DEV PERFERMANCE "+"="*6)
        dev_perf = sc.perf_eva(self.dev_woe[self.target], dev_pred, title="DEV")
        print("=" * 6 + " OOT PERFERMANCE " + "=" * 6)
        oot_perf = sc.perf_eva(self.oot_woe[self.target], oot_pred, title="OOT")

        orig_cols = [x[:-4] for x in self.input_cols]
        self.card = sc.scorecard(self.bins, lr, self.train[orig_cols].columns)
        dev_score = sc.scorecard_ply(self.train, self.card, print_step=0)
        oot_score = sc.scorecard_ply(self.test, self.card, print_step=0)
        # PSI
        sc.perf_psi(
            score={'train': dev_score, 'test': oot_score},
            label={'train': self.train[self.target], 'test': self.test['del_30']}
        )
        return self

    def mdoel_monitor(self):
        return self
