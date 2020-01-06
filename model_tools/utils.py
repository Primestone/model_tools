# -*- coding: utf-8 -*-

"""
utils
"""

import time
import numpy as np
import pandas as pd
import pickle
from contextlib import contextmanager


def is_numpy(x):
    return isinstance(x, np.ndarray)


def is_pandas(x):
    return isinstance(x, pd.DataFrame)


def get_rank(x):
    return pd.Series(x).rank(pct=True)


def get_age(idcard):
    current = int(time.strftime("%Y"))
    year = int(idcard[6:10])
    return current - year


def get_sex(idcard):
    try:
        if int(idcard[-2]) % 2 == 0:
            return 1
        else:
            return 0
    except:
        return 0


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[%s] done in %.2f s' % (name, time.time() - t0))


def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col]

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def replace_outline_category(test, train_value_dict, categorical_features=None, replace=True):
    value_diff_dict = {}
    if categorical_features is None:
        categorical_features = train_value_dict.keys()
    for col in categorical_features:
        test_cat_value = test[col].value_counts(dropna=False).index.tolist()
        value_diff = [x for x in test_cat_value if x not in train_value_dict[col]]
        if len(value_diff) > 0:
            value_diff_dict[col] = value_diff
        if replace:
            test.loc[test[col].isin(value_diff), col] = train_value_dict[col][0]
    return test


def save_pkl(file, save_name):
    pickle.dump(file, open("{}".format(save_name), "wb"))


def load_pkl(file_name):
    file = pickle.load(open(file_name, "rb"))
    return file


def p2c(p):
    if p<=0.1:
        return '[0, 0.1]'
    elif p<=0.2:
        return '(0.1, 0.2]'
    elif p<=0.3:
        return '(0.2, 0.3]'
    elif p<=0.4:
        return '(0.3, 0.4]'
    elif p<=0.5:
        return '(0.4, 0.5]'
    elif p<=0.6:
        return '(0.5, 0.6]'
    elif p<=0.7:
        return '(0.6, 0.7]'
    elif p<=0.8:
        return '(0.7, 0.8]'
    elif p<=0.9:
        return '(0.8, 0.9]'
    elif p<=1:
        return '(0.9, 1.0]'


def group_static(result, use_rank=False):
    if use_rank:
        result['prob'] = get_rank(result['prob'])
    
    result['pred_class'] = result['prob'].apply(p2c)
    res = pd.crosstab(result['pred_class'], result['target']).reset_index().rename(columns={0: 'Goods',1:'Bads'})
    res['Obs'] = res['Goods'] + res['Bads']
    res['Bad_Rate'] = res['Bads']/res['Obs']
    res['Cum_Bads'] = res['Bads'].cumsum()
    res['Cum_Goods'] = res['Goods'].cumsum()
    res['Cum_Bads_Pct'] = res['Cum_Bads']/res['Cum_Bads'].max()
    res['Cum_Goods_Pct'] = res['Cum_Goods']/res['Cum_Goods'].max()
    res['KS'] = abs(res['Cum_Bads_Pct'] - res['Cum_Goods_Pct'])*100
    return res

