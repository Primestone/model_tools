# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:32:41 2019

@author: Moon
"""

import pandas as pd
import numpy as np
import json
import model_helper
from functools import partial
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from lightgbm.sklearn import LGBMClassifier

from model_tools import pipeline
from model_tools.Feature_Engineer import groupby_stat, time_relation
from model_tools.FeatureSelector.select_utils import GreedyFeatureSelection
from model_tools import metrics
from model_tools.Model.params_tune import BayesOptim
from model_tools.Model.model_utils import KfoldClassifier, GreedyThresholdSelector
from model_tools.Preprocessing import encoders
from model_tools.Preprocessing.stabler import get_trend_stats
from model_tools.data import DataHelper
from model_tools.utils import *


# Extract Data
#hive_engine = sqlalchemy.create_engine('presto://10.8.49.170:8866/hive/default')
#
#sql = """select * from
#        (select a.*,m.dt_date from neo4j.xxxf_repair_8_1 a 
#         left join(
#         SELECT client_idno,dt_date,oc_phone_num from
#         (SELECT b.client_idno,b.dt_date,b.oc_phone_num, row_number() over(partition by b.client_idno,b.oc_phone_num order by b.dt_date asc ) as row_num
#          FROM neo4j.xxxf_repair_3_1 b)t where t.row_num=1)m on a.client_idno=m.client_idno and a.oc_phone_num=m.oc_phone_num
#         where m.client_idno is not null and m.oc_phone_num is not null)t """      
#                 
#info_data = pd.read_sql(sql, hive_engine)
#info_data.to_pickle('./input/repair_info_data.pkl')

#data = pd.read_pickle('./input/repair_info_data.pkl')
#data = data[data.call_type != 'phonebook']
#data['dt_date'] = pd.to_datetime(data['dt_date'])
#data['dt_date_month'] = data['dt_date'].apply(lambda x: str(x)[:7])
#
## data['dt_date_month'].value_counts()
#dev_sample = data[(data['dt_date'] >= '2017-10-01')&(data['dt_date'] < '2017-12-01')]
#oot_sample = data[(data['dt_date'] >= '2017-12-01')&(data['dt_date'] <= '2017-12-31')]
#
#dev_idno = list(dev_sample['client_idno'].drop_duplicates())
#oot_sample = oot_sample[~oot_sample['client_idno'].isin(dev_idno)].reset_index(drop=True)
#dev_sample.to_pickle('./input/dev_sample.pkl')
#oot_sample.to_pickle('./input/oot_sample.pkl')

with timer("- Sample Split."):
    dev_sample = pd.read_pickle('./input/dev_sample.pkl').reset_index(drop=True)
    oot_sample = pd.read_pickle('./input/oot_sample.pkl').reset_index(drop=True)
    print(f"- Dev sample size: {dev_sample.shape[0]}")
    print(f"- Oot sample size: {oot_sample.shape[0]}")
    
    unused_variables = ['mobile', 'oc_phone_num', 'oc_date', 'create_time', 'dt_date_month']
    dev_sample.drop(unused_variables, axis=1, inplace=True)
    oot_sample.drop(unused_variables, axis=1, inplace=True)
    
with timer("- Prepare Dataset."):
    datahper = DataHelper(target='is_valid', train_path=None, test_path=None,
                          trainfile=dev_sample, testfile=oot_sample,
                          date_cols=['dt_date'])
    data = datahper.combine()
    cat_vars = datahper.object_features
    date_vars = datahper.date_cols
    target = datahper.target

    data['age'] = data['client_idno'].apply(lambda x: 2018 - int(str(x)[6:10]))
    data['sex'] = data['client_idno'].apply(get_sex)
    data['call_in_ratio'] = data['call_in_len']/(data['call_len']+0.01)
    data['call_out_ratio'] = data['call_out_len']/(data['call_len']+0.01)
    data['total_contact_3m'] = data[['contact_1w', 'contact_1m','contact_3m']].apply(lambda x : sum(x), axis = 1)

with timer("- Pipeline Process."):   
    agg_func = {'client_idno': {'call_len': ['mean', 'std']},
                'client_idno#1': {'call_out_len_rt': ['mean', 'std']},
                'client_idno#2': {'call_in_cnt': ['mean', 'max', 'min', 'std']},
                'client_idno#3': {'total_contact_3m': ['mean', 'max', 'std', 'min']},
                'client_idno#4': {'relation_name': ['count', 'nunique']},
                'client_idno#5': {'contact_1w': ['mean', 'max', 'min', 'std']},
                'client_idno#6': {'contact_holiday': ['mean', 'max', 'min', 'std']},              
                'client_idno#7': {'call_type': ['count']},
                'client_idno#8': {'contact_all_day': ['count', 'mean']},
                
                'relation_name': {'call_len': ['mean', 'std']},
                'relation_name#1': {'call_out_len_rt': ['mean', 'std']},
                'relation_name#2': {'call_in_cnt': ['mean', 'max', 'min', 'std']},
                'relation_name#3': {'total_contact_3m': ['mean', 'max', 'std', 'min']},
                'relation_name#4': {'relation_name': ['count', 'nunique']},
                'relation_name#5': {'contact_1w': ['mean', 'max', 'min', 'std']},
                'relation_name#6': {'contact_holiday': ['mean', 'max', 'min', 'std']},
                'relation_name#7': {'call_type': ['count']},
                'relation_name#8': {'contact_all_day': ['count', 'mean']},
                }
    
    pipe = Pipeline([
                     ('CatgoryEncoder', encoders.CategoryEncoder(cat_vars)),
                     ('CountEncoder', encoders.CountEncoder(cat_vars)),
                     #('GroupbyStatic', groupby_stat.GroupbyStaticMethod(agg_func)),
                     #('ProcTime', time_relation.GentimerelatedFeaures(date_vars)),
                     ('DropConstant', pipeline.ConstantDropper())
                     ])

    data = pipe.fit_transform(data) 
    data['count_relation_rt'] = data['count_relation_name_by_client_idno']/data['cnt_enc_client_idno']
    data['unique_relation_rt'] = data['nunique_relation_name_by_client_idno']/data['cnt_enc_relation_name']

with timer("- Stabled Features."):
    dev_sample, oot_sample = datahper.split(data)
    data_summary = model_helper.data_report(dev_sample)
    #data_summary.to_csv('data_report.csv', index=False)
    
    variable_filter = data_summary[data_summary['unique'] > 1]
    variable_filter = variable_filter[~((variable_filter['dtype'] == 'object') & (variable_filter['unique'] >= 20))]
    variable_filter = variable_filter[variable_filter['missing_rate'] != 1]
    variable_filter = variable_filter[variable_filter['dtype'] != 'datetime64[ns]']
    
    use_cols = list(variable_filter['column'])
    use_cols.remove('client_idno')
    #stats = get_trend_stats(data=dev_sample[use_cols], target_col=target, data_test=oot_sample[use_cols])
    #stats.sort_values('Trend_correlation', ascending=False, inplace=True)
    #stats.to_excel('repair_info_stats.xlsx', index=False)
    stats = pd.read_excel('repair_info_stats.xlsx')

with timer("- Best Performance."):
    gbm_model = LGBMClassifier(boosting_type='gbdt', num_leaves=2**5, max_depth=5,
                     learning_rate=0.1, n_estimators=10000, #class_weight=20,
                     min_child_samples=20,
                     subsample=0.95, colsample_bytree=0.95,
                     reg_alpha=0.1, reg_lambda=0.1, random_state=1001,
                     sample_weight=None, #init_score=0.5
                     )

    result = GreedyThresholdSelector(dev_sample[use_cols], target, oot_sample[use_cols], gbm_model, stats, [0.78, 0.9], [1001, 925])
    result.to_excel("GreedysearchResult.xlsx", index=False)
    sel_cols = ['count_contact_all_day_by_client_idno', 'std_call_len_by_client_idno', 'mean_call_out_len_rt_by_client_idno', 'contact_1w_div_mean_contact_1w_by_client_idno', 'std_contact_1w_by_client_idno', 'contact_1m', 'total_contact_3m_div_mean_total_contact_3m_by_client_idno', 'contact_all_day_div_mean_contact_all_day_by_client_idno', 'call_out_ratio', 'std_total_contact_3m_by_client_idno', 'contact_3m_plus', 'std_call_in_cnt_by_client_idno', 'std_contact_holiday_by_client_idno', 'call_len_div_mean_call_len_by_client_idno', 'contact_night', 'contact_holiday_div_mean_contact_holiday_by_client_idno', 'mean_contact_holiday_by_client_idno', 'contact_afternoon', 'mean_call_len_by_client_idno', 'mean_contact_1w_by_client_idno', 'max_call_in_cnt_by_client_idno', 'call_in_cnt_dif_mean_call_in_cnt_by_client_idno', 'call_in_len_rank']
    _, oot_predict, oof_predict, _, _ = KfoldClassifier(dev_sample[sel_cols+[target]], target, oot_sample[sel_cols+[target]], gbm_model, seed=512, verbose=True, save_model=True) 
    print(f'- Test AUC: ', round(roc_auc_score(oot_sample[target], oot_predict), 4))
    print(f'- Test KS : ', round(metrics.ks(oot_sample[target], oot_predict), 4))
    #- OOF Train Auc: [0.6865], STD: [0.0023]
    #- OOF Train KS : [0.2619]
    #- Test AUC:  0.6815
    #- Test KS:  0.2529
 
# with timer("- Greedy Selector."):
#    sel_cols = stats[stats['Trend_correlation'] >= 0.9]['Feature'].tolist() + [target]
#    _, _, _, _, feature_imp = KfoldClassifier(dev_sample[sel_cols], target, oot_sample[sel_cols], gbm_model, n_folds=5, use_stratified=True, seed=1001, verbose=True)
#    top_cols = feature_imp['variables'].tolist()[:70] + [target]
#    FeatureSelector = GreedyFeatureSelection(dev_sample[top_cols], target, gbm_model, good_features=[], verbose=True)
#    good_features = FeatureSelector.selectionLoop()
#    save_pkl(good_features, 'GoodFeatures')
  
with timer("- Paramters Tuned."):
    good_features = ['call_len_div_mean_call_len_by_client_idno', 'total_contact_3m_div_mean_total_contact_3m_by_relation_name', 'count_contact_all_day_by_client_idno', 'contact_1w_div_mean_contact_1w_by_relation_name', 'contact_3m_plus', 'mean_call_out_len_rt_by_client_idno', 'sex', 'mean_contact_all_day_by_client_idno', 'contact_1m', 'mean_total_contact_3m_by_client_idno', 'max_call_in_cnt_by_client_idno', 'std_contact_1w_by_client_idno', 'unique_relation_rt', 'mean_call_len_by_client_idno', 'mean_contact_1w_by_client_idno', 'contact_all_day_dif_mean_contact_all_day_by_client_idno', 'call_out_cnt', 'max_total_contact_3m_by_client_idno', 'mean_call_in_cnt_by_client_idno', 'contact_3m', 'std_total_contact_3m_by_client_idno', 'contact_night', 'call_in_cnt_dif_mean_call_in_cnt_by_relation_name', 'max_contact_holiday_by_client_idno', 'min_total_contact_3m_by_client_idno', 'min_call_in_cnt_by_client_idno', 'call_len_div_mean_call_len_by_relation_name', 'relation_degree', 'max_contact_1w_by_client_idno', 'mean_contact_holiday_by_client_idno', 'call_cnt']
    sel_cols = good_features[:28]
    _, oot_predict, oof_predict, _, _ = KfoldClassifier(dev_sample[sel_cols+[target]], target, oot_sample[sel_cols+[target]], gbm_model, seed=1001, verbose=True, save_model=False)
    print(f'- Test AUC: ', round(roc_auc_score(oot_sample[target], oot_predict), 4)) 
    print(f'- Test KS : ', round(metrics.ks(oot_sample[target], oot_predict), 4))  
    #- OOF Train Auc: [0.6919], STD: [0.0016]
    #- OOF Train KS : [0.272]   
    #- Test AUC:  0.6866
    #- Test KS :  0.2622
    
    # parameters optim
    best_params = BayesOptim(train_df=dev_sample[sel_cols+[target]], target=target, init_points=5, n_iter=15)
    best_params = {'colsample_bytree': 0.8095258136489233,
                   'learning_rate': 0.07690519222771387,
                   'max_depth': 6,
                   'min_child_samples': 49,
                   'min_split_gain': 0.02896307396369064,
                   'n_estimators': 10000,
                   'nthread': 4,
                   'num_leaves': 16,
                   'reg_alpha': 0.265995923203365,
                   'reg_lambda': 0.26659431592624905,
                   'subsample': 0.9407857364187412}
    
    gbm_model = LGBMClassifier(**best_params)
    _, oot_predict, oof_predict, _, _ = KfoldClassifier(dev_sample[sel_cols+[target]], target, oot_sample[sel_cols+[target]], gbm_model, seed=1001, verbose=True, save_model=True)
    print(f'- Optim Test AUC: ', round(roc_auc_score(oot_sample[target], oot_predict), 4))
    print(f'- Optim Test KS : ', round(metrics.ks(oot_sample[target], oot_predict), 4))
