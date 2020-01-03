## Machine Learning Model Tools  :memo:
---
### BaseModule
**data**
 - DataHelper(target, train_path, test_path, trainfile=None, testfile=None, date_cols=None)
   [数据读取类，指定目标变量，训练测试数据的路径（或者加载好的文件），日期类型变量]
   可直接识别出离散型变量，日期变量，并支持train, test的合并、拆分
	 
**utils**
 - timer
 	 统计各个模块耗时的函数
 - reduce_mem_usage
	通过改变字段类型节约空间占用的函数
	
**pipeline**
 - ColumnSelector
 	支持变量的选择
 - ColumnDroper
   支持变量的删除
 - Dropconstant
   删除常量型的变量
 
**metrics**
  - ks(xgb_ks, lgb_ks)
  - rmse
  - gini
  ...
	
**estimators**
 - LikelihoodEstimatorUnivariate
 
    [**Target Encode Introduce**](http://www.saedsayad.com/encoding.htm)
		
 - LikelihoodEstimator

### Preprocessing
 - CategoryEncoder
   对离散型变量进行LabelEncoder
 - CountEncoder
   对离散型变量进行次数统计构造新变量
 - LikelihoodEncoder
   同上，target-based Encoding
 - PercentileEncoder
   [**连续性变量的ECDF转换**](http://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html)
 - NaEncoder
   缺失值的填充处理
 - Scaler
   数据的标准化处理
 - DummyEncoder
   基于离散特征构造哑变量
   
 - stabler
   测试变量在train、test上的稳定性工具

### Freature Engineer
 - FeatureCombiner
   离散型变量的组合（两个或多个）
 - GentimerelatedFeaures
   基于时间类型变量生成新的特征（year, month, weekday, hour, daysfromnow）
 - GroupbyStaticMethod
   基于离散型变量对连续型变量聚合得到新的统计特征（max, min, mean, count, sum, std, unique）
 - GBMEncoder
   GBDT每棵树的路径经one-hot-encode处理后生成新的特征
 
### Model
 - DNN
   深层的神经网络
 - xgboost-kfold
   支持K out-of-folds的内置xgboost分类器
 - lightgbm-kfold
   支持K out-of-folds的内置lightgbm分类器
 - catboost
   支持K out-of-folds的内置catboost分类器
 - params_tune
   基于贝叶斯优化的模型调参
 - model_utils
   n-folds classifier 和特征子集选择的贪心搜索
 - model_parser
   LightGBM模型文件解析工具
   
### Feature Selector
 - GreedyFeatureSelection
   step-wise逐步特征选择（可基于所有分类器）
 - BaseFeatureImportance
   给定阈值，基于特征重要性的特征选择方法
 - select_utils
   基于指定分类器的向前（向后）特征递归选择
 
### Ensemble
 - Stacking
   模型融合
	 
	 ![image](https://img-blog.csdn.net/20170915114447314?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3N0Y2pm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

model1: 将数分层5折，4份做训练，剩余1份做交叉验证， 训练5次，这样会得到一个完整train的new feature, 预测的test的5个结果取均值，得到test的new feature,这样就得到model1的meta feature1。同理，model2按照该过程得到meta feature2, n个模型可得到n个新的特征，作为下一层模型的输入，经训练后得到预测结果。

### Information Repair Demo Structure
基于以上工具，以信修数据为建模样本，更新一版模型
---
```python
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
                     ('GroupbyStatic', groupby_stat.GroupbyStaticMethod(agg_func)),
                     ('ProcTime', time_relation.GentimerelatedFeaures(date_vars)),
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

#    result = GreedyThresholdSelector(dev_sample[use_cols], target, oot_sample[use_cols], gbm_model, stats, [0.78, 0.9], [1001, 925])  
#    result.to_excel("GreedysearchResult.xlsx", index=False)
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

```
---

#### model perfermence (auc、ks)
algorithm| dataset  |  5fold-cv | test
----- | ------ | ------ | -----
lgbm_v2 | info-repair | auc(0.692) ks(27.2) | auc(0.687) ks(26.2)
