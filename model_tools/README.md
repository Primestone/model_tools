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
  - auc
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
   基于时间类型变量生成新的特征（year, month, weekday, hour, days from now）
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
   n-folds classifier和特征子集选择的贪心搜索
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


### Test Demo
---
```python

import sys
sys.path.append('..')

import pandas as pd
from sklearn.pipeline import Pipeline
from model_tools.Feature_Engineer.time_relation import GentimerelatedFeaures
from model_tools.AutoModel import AutoXGBoost
from model_tools.metrics import roc_auc_score, ks


class AutoXGBoost_BaseBR(AutoXGBoost):

    def _data_preprocess(self):
        tl_date_variables = ['tl_cell_lasttime', 'tl_id_lasttime']
        date_pipe = Pipeline([
                ("ProcTime", GentimerelatedFeaures(tl_date_variables)),
                ])
    
        self.train = date_pipe.fit_transform(self.train)
        self.test = date_pipe.fit_transform(self.test)
        
        drop_type = ['_year', '_month', '_doy', '_dow']
        for col in tl_date_variables:
            drop_cols = [col + _type for _type in drop_type]
            self.train.drop(drop_cols, axis=1, inplace=True)
            self.test.drop(drop_cols, axis=1, inplace=True)


model_data = pd.read_csv('./input/model_data.csv')
key = 'appl_no'
target = 'fpd10'

unuse_cols = ['flagJgV2',
              'brF001flag',
              'loan_date',       
              #'fpd10'
              'spd15',
              'del_10',
              'del_30',
              'fipd30'
              ]

model_data = model_data[model_data[target].notnull()]
dev_sample = model_data[model_data['loan_date']<='2019-05-10'].reset_index(drop=True)
oot_sample = model_data[model_data['loan_date']>'2019-05-10'].reset_index(drop=True)

print("DEV SAMPLE SIZE: %d, BAD RATIO: %.4f" %
              (dev_sample.shape[0], dev_sample[target].mean()))
print("OOT SAMPLE SIZE: %d, BAD RATIO: %.4f" %
      (oot_sample.shape[0], oot_sample[target].mean()))

autoxgb = AutoXGBoost_BaseBR(
        train=dev_sample,
        test=oot_sample,
        target=target,
        key='appl_no',
        unuse_variables=unuse_cols,
        categorical_features = None,
        date_cols=[],
        missing_rate_threshold=0.85,
        select_min=30,
        max_features_num=120,
        trend_correlation_list=[.7, -1],
        output_file='./output/',
        project_name='Device_Fraud_Model',
        bin_method='tree',
        bin_num=7
        )
autoxgb._feature_select()
autoxgb.xgboost_model()
```

基于以上自动建模模块，会在output文件夹下自动生成数据报告, 数据预测量pipeline, 训练样本及测试样本上变量的
一致性统计, 基于shap值计算的特征重要性, 贪心选择模型表现, 选择的变量集, 模型文件, 分数分组统计结果.
