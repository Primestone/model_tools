# -*- coding: utf-8 -*-
"""
score card model helper
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as sm_vif
import matplotlib.pyplot as plt
import datetime
import time
import itertools
import math
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import os

def data_report(in_data, numeric_missing_value = np.nan, string_missing_value = ''):
    desc = in_data.describe(percentiles = [0.25, 0.5, 0.75, 0.95, 0.99], include = 'all').transpose().reset_index()
    
    if 'unique' in desc.columns:
        desc = desc.drop(['unique', 'top', 'freq'], axis = 1)
    
    desc = desc.rename(columns = {'index': 'column'})
    
    num_missing_to_1 = in_data.select_dtypes(include = ['number']).applymap(
        lambda x: 1 if np.isnan(x) or x == numeric_missing_value else 0)
    count_num_missing = num_missing_to_1.apply(np.sum, axis = 0).reset_index().rename(columns = {'index': 'column', 0: 'missing'})
    
    str_missing_to_1 = in_data.select_dtypes(include = ['object']).applymap(lambda x: 1 if x == string_missing_value else 0)
    count_str_missing = str_missing_to_1.apply(np.sum, axis = 0).reset_index().rename(columns = {'index': 'column', 0: 'missing'})
    
    datetime_missing_to_1 = in_data.select_dtypes(include = ['datetime']).applymap(lambda x: 1 if x == None else 0)
    count_datetime_missing = datetime_missing_to_1.apply(np.sum, axis = 0).reset_index().rename(
        columns = {'index': 'column', 0: 'missing'})
    
    count_missing = pd.concat([count_num_missing, count_str_missing, count_datetime_missing])
    
    column_unique = in_data.apply(pd.Series.nunique, axis = 0).reset_index().rename(columns = {'index': 'column', 0: 'unique'})
    desc = desc.merge(column_unique, how = 'left', on = 'column')
    
    column_types = pd.DataFrame(in_data.dtypes).reset_index().rename(columns = {'index': 'column', 0: 'dtype'})
    desc = desc.merge(column_types, how = 'left', on = 'column').merge(count_missing, how = 'left', on = 'column')
    
    desc['count'] = in_data.index.size
    desc['missing_rate'] = desc['missing'] / desc['count']
    
    if 'first' in desc.columns:
        desc['first'] = desc['first'].fillna('')
        desc['last'] = desc['last'].fillna('')
    
    desc['count'] = desc['count'].astype(np.int)
    
    if 'mean' in desc.columns:
        desc['mean'] = desc['mean'].astype(np.float)
        desc['std'] = desc['std'].astype(np.float)
        desc['min'] = desc['min'].astype(np.float)
        desc['25%'] = desc['25%'].astype(np.float)
        desc['50%'] = desc['50%'].astype(np.float)
        desc['75%'] = desc['75%'].astype(np.float)
        desc['95%'] = desc['95%'].astype(np.float)
        desc['99%'] = desc['99%'].astype(np.float)
        desc['max'] = desc['max'].astype(np.float)
    desc['missing_rate'] = desc['missing_rate'].astype(np.float)
    desc['dtype'] = desc['dtype'].astype(str)
    
    if 'first' in desc.columns:
        desc['first'] = desc['first'].astype(str)
        desc['last'] = desc['last'].astype(str)
    
    if 'first' in desc.columns:
        desc = desc[['column', 'dtype', 'count', 'missing', 'missing_rate', 'unique', 'mean', 
                     'std', 'min', '25%', '50%', '75%', '95%', '99%', 'max', 'first', 'last']]
    elif 'mean' in desc.columns:
        desc = desc[['column', 'dtype', 'count', 'missing', 'missing_rate', 'unique', 'mean', 
                     'std', 'min', '25%', '50%', '75%', '95%', '99%', 'max']]
    else:
        desc = desc[['column', 'dtype', 'count', 'missing', 'missing_rate', 'unique']]
    return desc


def create_woe(in_data, bin_variable, target_variable, woe_suffix = '_woe', num_bins = 10, return_woe_appended_data = True, test=None):
    calc_data = in_data[[target_variable, bin_variable]].copy()
    bin_name = bin_variable + '_bin_group_n'
    unique_size = calc_data[bin_variable].unique().size
    
    if pd.api.types.is_object_dtype(calc_data[bin_variable].dtype) and unique_size >= 20:
        print('Warning: too many unique values for {0}'.format(bin_variable))
    
    if pd.api.types.is_numeric_dtype(calc_data[bin_variable].dtype) and unique_size >= 20:
        try:
            calc_data[bin_name] = pd.qcut(calc_data[bin_variable], num_bins, labels = False, duplicates = 'drop').rename(bin_name)
        except:
            try:
                calc_data[bin_name] = pd.qcut(calc_data[bin_variable], num_bins, labels = False).rename(bin_name)
            except:
                calc_data[bin_name] = pd.cut(calc_data[bin_variable], num_bins, labels = False).rename(bin_name)
        
        calc_data['min'] = calc_data[bin_variable]
        calc_data['max'] = calc_data[bin_variable]
        bins_min_max = calc_data.groupby(bin_name, as_index = False).agg({'min': np.min, 'max': np.max})
        bins_min_max['min_str'] = bins_min_max['min'].apply(lambda x: str(int(x)) if float.is_integer(float(x)) else str(x))
        bins_min_max['max_str'] = bins_min_max['max'].apply(lambda x: str(int(x)) if float.is_integer(float(x)) else str(x))
        bins_min_max['value'] = (bins_min_max[bin_name] + 1).astype(np.int64).astype(str).str.zfill(len(str(num_bins))) + \
            '. [' + bins_min_max['min_str'] + ', ' + bins_min_max['max_str'] + ']'
        
        del calc_data['min'], calc_data['max'], bins_min_max['min'], bins_min_max['max']
        calc_data = calc_data.merge(bins_min_max, how = 'left', on = bin_name)
        calc_data['value'].fillna('{0}. NaN'.format('0'.zfill(len(str(num_bins)))), inplace = True)
        del calc_data[bin_name], bins_min_max
    else:
        calc_data['value'] = calc_data[bin_variable]
    
    calc_data['cnt_bad'] =  np.where(calc_data[target_variable] == 1, 1, 0)
    calc_data['cnt_good'] = np.where(calc_data[target_variable] == 0, 1, 0)
    
    woe_table = calc_data.groupby('value', as_index = False).agg({'cnt_bad': np.sum, 'cnt_good': np.sum})
    
    nan_data = calc_data[calc_data['value'].isnull()]
    if nan_data.empty == False:
        woe_nan = {}
        woe_nan['value'] = np.nan
        woe_nan['cnt_bad']  = nan_data['cnt_bad'].sum()
        woe_nan['cnt_good'] = nan_data['cnt_good'].sum()
        
        woe_table.index += 1
        woe_table = pd.concat([pd.DataFrame(woe_nan, index = [0]), woe_table])[['value', 'cnt_bad', 'cnt_good']]
    
    woe_table['variable']       = bin_variable
    woe_table['bad_rate']       = woe_table['cnt_bad'] / (woe_table['cnt_bad'] + woe_table['cnt_good'])
    woe_table['cnt_bad_total']  = woe_table['cnt_bad'].sum()
    woe_table['cnt_good_total'] = woe_table['cnt_good'].sum()
    woe_table['pct_bad']        = woe_table['cnt_bad'] / woe_table['cnt_bad_total']
    woe_table['pct_good']       = woe_table['cnt_good'] / woe_table['cnt_good_total']
    
    with np.errstate(divide = 'ignore'):
        woe_table['woe'] = \
            np.where(woe_table['pct_bad']  == 0,  10e8, 
            np.where(woe_table['pct_good'] == 0, -10e8, np.log(woe_table['pct_good'] / woe_table['pct_bad']) * 100))
    
    calc_data = calc_data.merge(woe_table[['value', 'woe']], how = 'left', on = 'value')
    
    if return_woe_appended_data == True:
        calc_data.index = in_data.index
        in_data[bin_variable + '_bin'] = calc_data['value'].astype(str)
        in_data[bin_variable + woe_suffix] = calc_data['woe']
    
    del bin_name, bin_variable, calc_data, num_bins, target_variable, unique_size, woe_suffix, nan_data
    
    woe_table['value'] = woe_table['value'].apply(lambda x: 
        str(int(x)) if pd.api.types.is_number(x) and float.is_integer(float(x)) else str(x))
    
    rows_regular_woe = woe_table[~woe_table['bad_rate'].isin([0, 1])].copy().drop(columns = ['variable'])
    rows_extreme_woe = woe_table[ woe_table['bad_rate'].isin([0, 1])].copy().drop(columns = ['variable'])
    
    if rows_extreme_woe.empty:
        woe_table['woe_grouped']        = woe_table['value']
        woe_table['cnt_bad_new']        = woe_table['cnt_bad']
        woe_table['cnt_good_new']       = woe_table['cnt_good']
        woe_table['bad_rate_new']       = woe_table['bad_rate']
        woe_table['cnt_bad_total_new']  = woe_table['cnt_bad_total']
        woe_table['cnt_good_total_new'] = woe_table['cnt_good_total']
        woe_table['pct_bad_new']        = woe_table['pct_bad']
        woe_table['pct_good_new']       = woe_table['pct_good']
        woe_table['woe_new']            = woe_table['woe']
        woe_table['iv'] = ((woe_table['pct_good_new'] - woe_table['pct_bad_new']) * woe_table['woe_new'] / 100).sum()
        
        woe_table = woe_table[['variable', 'value', 'cnt_bad', 'cnt_good', 'bad_rate', 'cnt_bad_total', 'cnt_good_total', 
                               'pct_bad', 'pct_good', 'woe', 'woe_grouped', 'cnt_bad_new', 'cnt_good_new', 'bad_rate_new', 
                               'cnt_bad_total_new', 'cnt_good_total_new', 'pct_bad_new', 'pct_good_new', 'woe_new', 'iv']]
        
        if return_woe_appended_data == True:
            return in_data, woe_table
        else:
            return woe_table
    
    combined_extreme_woe = pd.DataFrame()
    
    adjacent_index_list = []
    for i in range(rows_extreme_woe.index.size):
        if rows_extreme_woe.index[i] in adjacent_index_list:
            continue
        else:
            adjacent_index_list.clear()
            adjacent_index_list.append(rows_extreme_woe.index[i])
        
        for j in range(i + 1, rows_extreme_woe.index.size):
            if rows_extreme_woe.index[j] == adjacent_index_list[-1] + 1:
                adjacent_index_list.append(rows_extreme_woe.index[j])
            else:
                continue
        
        adjacent_rows = rows_extreme_woe[rows_extreme_woe.index.isin(adjacent_index_list)]
        combined = {}
        combined['value']          = ' | '.join(adjacent_rows['value'])
        combined['cnt_bad']        = adjacent_rows['cnt_bad'].sum()
        combined['cnt_good']       = adjacent_rows['cnt_good'].sum()
        combined['bad_rate']       = combined['cnt_bad'] / (combined['cnt_bad'] + combined['cnt_good'])
        combined['cnt_bad_total']  = adjacent_rows['cnt_bad_total'].mean()
        combined['cnt_good_total'] = adjacent_rows['cnt_good_total'].mean()
        combined['pct_bad']        = combined['cnt_bad'] / combined['cnt_bad_total']
        combined['pct_good']       = combined['cnt_good'] / combined['cnt_good_total']
        
        with np.errstate(divide = 'ignore'):
            combined['woe'] = 10e8 if combined['pct_bad'] == 0 else -10e8 if combined['pct_good'] == 0 \
                else np.log(combined['pct_good'] / combined['pct_bad']) * 100
        
        combined_extreme_woe = combined_extreme_woe.append(pd.DataFrame(combined, index = [adjacent_rows.index.min()]))
    
    rows_regular_woe = rows_regular_woe.append(combined_extreme_woe).sort_index()
    still_extreme = rows_regular_woe[rows_regular_woe['bad_rate'].isin([0, 1])]
    
    for i in still_extreme.index:
        index_prev = rows_regular_woe[rows_regular_woe.index < i].index.max()
        index_next = rows_regular_woe[rows_regular_woe.index > i].index.min()
        row_prev = rows_regular_woe[rows_regular_woe.index == index_prev]
        row_next = rows_regular_woe[rows_regular_woe.index == index_next]
        
        if np.isnan(index_prev):
            index_merge = index_next
        elif np.isnan(index_next):
            index_merge = index_prev
        else:
            if abs(still_extreme.iloc[0]['woe'] - row_prev.iloc[0]['woe']) <= \
                abs(still_extreme.iloc[0]['woe'] - row_next.iloc[0]['woe']):
                index_merge = index_prev
            else:
                index_merge = index_next
        
        to_be_merged = pd.concat([rows_regular_woe[rows_regular_woe.index == index_merge], 
                                  still_extreme[still_extreme.index == i]]).sort_index()
        
        extreme_merged = {}
        extreme_merged['value']          = ' | '.join(to_be_merged['value'])
        extreme_merged['cnt_bad']        = to_be_merged['cnt_bad'].sum()
        extreme_merged['cnt_good']       = to_be_merged['cnt_good'].sum()
        extreme_merged['bad_rate']       = extreme_merged['cnt_bad'] / (extreme_merged['cnt_bad'] + extreme_merged['cnt_good'])
        extreme_merged['cnt_bad_total']  = to_be_merged['cnt_bad_total'].mean()
        extreme_merged['cnt_good_total'] = to_be_merged['cnt_good_total'].mean()
        extreme_merged['pct_bad']        = extreme_merged['cnt_bad'] / extreme_merged['cnt_bad_total']
        extreme_merged['pct_good']       = extreme_merged['cnt_good'] / extreme_merged['cnt_good_total']
        extreme_merged['woe']            = np.log(extreme_merged['pct_good'] / extreme_merged['pct_bad']) * 100
        
        rows_regular_woe.drop([i, index_merge], inplace = True)
        rows_regular_woe = rows_regular_woe.append(pd.DataFrame(extreme_merged, index = [to_be_merged.index.min()])).sort_index()
    
    rows_regular_woe.reset_index(drop = True, inplace = True)
    rows_regular_woe.rename(columns = {'value': 'woe_grouped', 'bad_rate': 'bad_rate_new', 'cnt_bad': 'cnt_bad_new', 
                                       'cnt_bad_total': 'cnt_bad_total_new', 'cnt_good': 'cnt_good_new', 
                                       'cnt_good_total': 'cnt_good_total_new', 'pct_bad': 'pct_bad_new', 
                                       'pct_good': 'pct_good_new', 'woe': 'woe_new'}, inplace = True)
    
    rows_regular_woe['iv'] = \
        ((rows_regular_woe['pct_good_new'] - rows_regular_woe['pct_bad_new']) * rows_regular_woe['woe_new'] / 100).sum()
    
    woe_table = woe_table.merge(rows_regular_woe, how = 'left', left_index = True, right_index = True)
    woe_table['iv'] = woe_table['iv'].fillna(woe_table['iv'].max())
    
    woe_table = woe_table[['variable', 'value', 'cnt_bad', 'cnt_good', 'bad_rate', 'cnt_bad_total', 'cnt_good_total', 
                           'pct_bad', 'pct_good', 'woe', 'woe_grouped', 'cnt_bad_new', 'cnt_good_new', 'bad_rate_new', 
                           'cnt_bad_total_new', 'cnt_good_total_new', 'pct_bad_new', 'pct_good_new', 'woe_new', 'iv']]
    
    del adjacent_index_list, adjacent_rows, combined, combined_extreme_woe, i, still_extreme, rows_extreme_woe, rows_regular_woe
    
    if return_woe_appended_data == True:
        return in_data, woe_table
    else:
        return woe_table


def create_woe_list(in_data, variable_woe_list, target_variable, num_bins = 10, return_woe_appended_data = True):
    woe_complete = pd.DataFrame()
    total_num = len(variable_woe_list)
    current_num = 1
    
    for i in variable_woe_list:
        print('WoE - ' + str(current_num) + ' of ' + str(total_num) + ': ' + i)
        
        if return_woe_appended_data == True:
            in_data, woe_result = create_woe(in_data, i, target_variable, '_woe', num_bins = num_bins)
        else:
            woe_result = create_woe(in_data, i, target_variable, '_woe', num_bins, return_woe_appended_data = False)
        woe_complete = woe_complete.append(woe_result)
        current_num += 1
    
    woe_complete.reset_index(inplace = True)
    woe_complete.sort_values(['iv', 'variable', 'index'], ascending = [False, True, True], inplace = True)
    
    if return_woe_appended_data == True:
        return in_data, woe_complete
    else:
        return woe_complete


def create_bin(in_data, variable, boundary_list, combine_missing_to = 'alone'):
    seq_num = 1
    zfill_len = len(str(len(boundary_list) + 2))
    boundary_left = -np.inf
    
    boundary_list.append(np.inf)
    
    out_series = pd.Series(index = in_data.index)
    
    for i in boundary_list:
        values_unique = in_data[(in_data[variable] > boundary_left) & (in_data[variable] <= i)][variable].unique()
        min_value = int(values_unique.min()) if float.is_integer(float(values_unique.min())) else values_unique.min()
        max_value = int(values_unique.max()) if float.is_integer(float(values_unique.max())) else values_unique.max()
        
        if values_unique.size == 1:
            bin_value = '{0}. [{1}]'.format(str(seq_num).zfill(zfill_len), str(min_value))
        else:
            if i == np.inf:
                bin_value = '{0}. [{1}, inf]'.format(str(seq_num).zfill(zfill_len), str(min_value))
            else:
                bin_value = '{0}. [{1}, {2}]'.format(str(seq_num).zfill(zfill_len), str(min_value), str(max_value))
        
        if combine_missing_to.lower() == 'top':
            if seq_num == 1:
                out_series = pd.Series(
                    np.where(((in_data[variable] > boundary_left) & (in_data[variable] <= i)) | in_data[variable].isnull(), 
                        bin_value + ' | NaN', out_series), index = in_data.index)
            else:
                out_series = pd.Series(
                    np.where((in_data[variable] > boundary_left) & (in_data[variable] <= i), bin_value, out_series), 
                        index = in_data.index)
        elif combine_missing_to.lower() == 'bottom':
            if seq_num == len(boundary_list):
                out_series = pd.Series(
                    np.where(((in_data[variable] > boundary_left) & (in_data[variable] <= i)) | in_data[variable].isnull(), 
                        bin_value + ' | NaN', out_series), index = in_data.index)
            else:
                out_series = pd.Series(
                    np.where((in_data[variable] > boundary_left) & (in_data[variable] <= i), bin_value, out_series), 
                        index = in_data.index)
        else:
            out_series = pd.Series(
                np.where((in_data[variable] > boundary_left) & (in_data[variable] <= i), bin_value, out_series), 
                    index = in_data.index)
            
            if seq_num == len(boundary_list):
                out_series = pd.Series(np.where(in_data[variable].isnull(), '0. NaN', out_series), index = in_data.index)
        
        seq_num += 1
        boundary_left = max_value
    
    print('Binning: {0}'.format(variable))
    print(out_series.value_counts(dropna = False).sort_index())
    print('')
    
    del bin_value, boundary_left, boundary_list, combine_missing_to, i, max_value, min_value, seq_num
    del values_unique, variable, zfill_len
    
    return out_series


def logit_fit(in_data, y_variable, x_variables, title = 'Fitting Logistic Regression Model', plot_text = ''):
    separator = '\n==================================================\n'
    print(separator + str(title) + separator)
    
    model_data = sm.add_constant(in_data[x_variables])
    
    logit_reg = sm.Logit(in_data[y_variable], model_data)
    try:
        result = logit_reg.fit()
    except:
        result = logit_reg.fit(method='bfgs')
    print(result.summary2())
    
    model_data = model_data.merge(in_data[[y_variable]], how = 'left', left_index = True, right_index = True)
    
    model_data['prob'] = logit_reg.predict(result.params)
    
    model_data['1_y'] = (1 - model_data[y_variable]).astype(np.float64)
    model_data_grouped = model_data[['prob', y_variable, '1_y']].groupby('prob').agg(
        {y_variable: np.sum, '1_y': np.sum}).reset_index()
    
    sensitivity = model_data_grouped.sort_values('prob', ascending = False)
    sensitivity['cum_sum'] = sensitivity[y_variable].cumsum()
    sensitivity = sensitivity.sort_values('cum_sum', ascending = False)['cum_sum'] / np.sum(model_data[y_variable])
    
    specificity = model_data_grouped.sort_values('prob')
    specificity['cum_sum'] = specificity['1_y'].cumsum()
    specificity = specificity['cum_sum'] / (model_data.shape[0] - np.sum(model_data[y_variable]))
    
    auc = (specificity.diff() * (sensitivity + sensitivity.shift().fillna(1))).sum() / 2
    
    model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (
        model_data.shape[0] - np.sum(model_data[y_variable]))
    model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(model_data[y_variable])
    model_data_grouped['ks'] = (model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100
    max_ks = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]
    
    plt.plot(1 - specificity, sensitivity, linewidth = 2)
    plt.plot([0, 1], [0, 1], color = 'grey')
    plt.text(0.7, 0.16, plot_text, fontsize = 14)
    plt.text(0.7, 0.1, 'RoC Curve', fontsize = 14)
    plt.text(0.7, 0.04, 'AUC = %.4f' % auc, fontsize = 14)
    plt.xlabel('1 - Specificity', size = 14)
    plt.ylabel('Sensitivity', size = 14)
    plt.show()
    print('AUC = %.4f' % auc)
    
    plt.plot(model_data_grouped['prob'], model_data_grouped['cum_pct_0'], linewidth = 2)
    plt.plot(model_data_grouped['prob'], model_data_grouped['cum_pct_1'], linewidth = 2)
    plt.plot([max_ks['prob'].iloc[0], max_ks['prob'].iloc[0]], 
             [max_ks['cum_pct_0'].iloc[0], max_ks['cum_pct_1'].iloc[0]], color = 'grey')
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.16, 
                                plot_text, fontsize = 14)    
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.1, 
                                'Prob. Distribution', fontsize = 14)
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.04, 
                                'KS = %.2f' % max_ks['ks'].iloc[0], fontsize = 14)
    plt.xlabel('Probability', size = 14)
    plt.ylabel('Proportion', size = 14)
    plt.show()
    print('KS = %.2f' % max_ks['ks'].iloc[0])
    
    return result.params


def logit_score(in_data, model_params, y_variable, x_variables, title = 'Scoring Data', plot_text = ''):
    separator = '\n==================================================\n'
    print(separator + str(title) + separator)
    
    model_data = sm.add_constant(in_data[x_variables])
    logit_reg = sm.Logit(in_data[y_variable], model_data)
    
    model_data = model_data.merge(in_data[[y_variable]], how = 'left', left_index = True, right_index = True)
    model_data['prob'] = logit_reg.predict(model_params)
    
    model_data['1_y'] = (1 - model_data[y_variable]).astype(np.float64)
    model_data_grouped = model_data[['prob', y_variable, '1_y']].groupby('prob').agg(
        {y_variable: np.sum, '1_y': np.sum}).reset_index()
    
    sensitivity = model_data_grouped.sort_values('prob', ascending = False)
    sensitivity['cum_sum'] = sensitivity[y_variable].cumsum()
    sensitivity = sensitivity.sort_values('cum_sum', ascending = False)['cum_sum'] / np.sum(model_data[y_variable])
    
    specificity = model_data_grouped.sort_values('prob')
    specificity['cum_sum'] = specificity['1_y'].cumsum()
    specificity = specificity['cum_sum'] / (model_data.shape[0] - np.sum(model_data[y_variable]))
    
    auc = (specificity.diff() * (sensitivity + sensitivity.shift().fillna(1))).sum() / 2
    
    model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (
        model_data.shape[0] - np.sum(model_data[y_variable]))
    model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(model_data[y_variable])
    model_data_grouped['ks'] = (model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100
    max_ks = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]
    
    plt.plot(1 - specificity, sensitivity, linewidth = 2)
    plt.plot([0, 1], [0, 1], color = 'grey')
    plt.text(0.7, 0.16, plot_text, fontsize = 14)
    plt.text(0.7, 0.1, 'RoC Curve', fontsize = 14)
    plt.text(0.7, 0.04, 'AUC = %.4f' % auc, fontsize = 14)
    plt.xlabel('1 - Specificity', size = 14)
    plt.ylabel('Sensitivity', size = 14)
    plt.show()
    print('AUC = %.4f' % auc)
    
    plt.plot(model_data_grouped['prob'], model_data_grouped['cum_pct_0'], linewidth = 2)
    plt.plot(model_data_grouped['prob'], model_data_grouped['cum_pct_1'], linewidth = 2)
    plt.plot([max_ks['prob'].iloc[0], max_ks['prob'].iloc[0]], 
             [max_ks['cum_pct_0'].iloc[0], max_ks['cum_pct_1'].iloc[0]], color = 'grey')
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.16, 
                                plot_text, fontsize = 14)
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.1, 
                                'Prob. Distribution', fontsize = 14)
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.04, 
                                'KS = %.2f' % max_ks['ks'].iloc[0], fontsize = 14)
    plt.xlabel('Probability', size = 14)
    plt.ylabel('Proportion', size = 14)
    plt.show()
    print('KS = %.2f' % max_ks['ks'].iloc[0])


def logit_draw(in_data, y_variable, title = '', plot_text = ''):
    separator = '\n==================================================\n'
    print(separator + str(title) + separator)
    
    model_data = in_data.copy()
    
    model_data['1_y'] = (1 - model_data[y_variable]).astype(np.float64)
    model_data_grouped = model_data[['prob', y_variable, '1_y']].groupby('prob').agg({
        y_variable: np.sum, '1_y': np.sum}).reset_index()
    
    sensitivity = model_data_grouped.sort_values('prob', ascending = False)
    sensitivity['cum_sum'] = sensitivity[y_variable].cumsum()
    sensitivity = sensitivity.sort_values('cum_sum', ascending = False)['cum_sum'] / np.sum(model_data[y_variable])
    
    specificity = model_data_grouped.sort_values('prob')
    specificity['cum_sum'] = specificity['1_y'].cumsum()
    specificity = specificity['cum_sum'] / (model_data.shape[0] - np.sum(model_data[y_variable]))
    
    auc = (specificity.diff() * (sensitivity + sensitivity.shift().fillna(1))).sum() / 2
    
    model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (model_data.shape[0] - np.sum(model_data[y_variable]))
    model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(model_data[y_variable])
    model_data_grouped['ks'] = (model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100
    max_ks = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]
    
    plt.plot(1 - specificity, sensitivity, linewidth = 2)
    plt.plot([0, 1], [0, 1], color = 'grey')
    plt.text(0.7, 0.16, plot_text, fontsize = 14)
    plt.text(0.7, 0.1, 'RoC Curve', fontsize = 14)
    plt.text(0.7, 0.04, 'AUC = %.4f' % auc, fontsize = 14)
    plt.xlabel('1 - Specificity', size = 14)
    plt.ylabel('Sensitivity', size = 14)
    plt.show()
    print('AUC = %.4f' % auc)
    
    plt.plot(model_data_grouped['prob'], model_data_grouped['cum_pct_0'], linewidth = 2)
    plt.plot(model_data_grouped['prob'], model_data_grouped['cum_pct_1'], linewidth = 2)
    plt.plot([max_ks['prob'].iloc[0], max_ks['prob'].iloc[0]], 
             [max_ks['cum_pct_0'].iloc[0], max_ks['cum_pct_1'].iloc[0]], color = 'grey')
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.16, 
                                plot_text, fontsize = 14)
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.1, 
                                'Prob. Distribution', fontsize = 14)
    plt.text(0.1 + 0.4 * np.max(model_data_grouped['prob']), np.min(model_data_grouped['cum_pct_1']) + 0.04, 
                                'KS = %.2f' % max_ks['ks'].iloc[0], fontsize = 14)
    plt.xlabel('Probability', size = 14)
    plt.ylabel('Proportion', size = 14)
    plt.show()
    print('KS = %.2f' % max_ks['ks'].iloc[0])


def logit_draw_plotly(in_data, y_variable, score_var, higher_better = True, title = '', plot_text = ''):
    import plotly
    separator = '\n==================================================\n'
    print(separator + str(title) + separator)
    
    model_data = in_data.copy()
    
    model_data['1_y'] = (1 - model_data[y_variable]).astype(np.float64)
    model_data_grouped = model_data[[score_var, y_variable, '1_y']].groupby(score_var).agg({
        y_variable: np.sum, '1_y': np.sum}).reset_index()
    
    sensitivity = model_data_grouped.sort_values(score_var, ascending = higher_better)
    sensitivity['cum_sum'] = sensitivity[y_variable].cumsum()
    sensitivity = sensitivity.sort_values('cum_sum', ascending = higher_better)['cum_sum'] / np.sum(model_data[y_variable])
    
    specificity = model_data_grouped.sort_values(score_var, ascending = (not higher_better))
    specificity['cum_sum'] = specificity['1_y'].cumsum()
    specificity = specificity.sort_values(
        'cum_sum', ascending = (not higher_better))['cum_sum'] / (model_data.shape[0] - np.sum(model_data[y_variable]))
    
    auc = abs((specificity.diff() * (sensitivity + sensitivity.shift().fillna(1))).sum() / 2)
    
    model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (model_data.shape[0] - np.sum(model_data[y_variable]))
    model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(model_data[y_variable])
    model_data_grouped['ks'] = abs((model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100)
    max_ks = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]
    
    line_cum_good = plotly.graph_objs.Scatter(
        name = 'Cum. Bad', x = model_data_grouped[score_var], y = model_data_grouped['cum_pct_0'])
    line_cum_bad = plotly.graph_objs.Scatter(
        name = 'Cum. Good', x = model_data_grouped[score_var], y = model_data_grouped['cum_pct_1'])
    line_max_ks = plotly.graph_objs.Scatter(name = 'Max KS', x = [max_ks[score_var].iloc[0], max_ks[score_var].iloc[0]], 
                                            y = [max_ks['cum_pct_0'].iloc[0], max_ks['cum_pct_1'].iloc[0]])
    
    line_roc = plotly.graph_objs.Scatter(name = 'ROC', x = 1 - specificity, y = sensitivity)
    line_diagonal = plotly.graph_objs.Scatter(name = 'Diagonal', x = [0, 1], y = [0, 1])
    
    figure = plotly.tools.make_subplots(rows = 1, cols = 2, print_grid = False, 
                                        subplot_titles = ['Model KS = %.2f' % max_ks['ks'].iloc[0], 'ROC AUC = %.4f' % auc])
    figure.append_trace(line_cum_good, 1, 1)
    figure.append_trace(line_cum_bad, 1, 1)
    figure.append_trace(line_max_ks, 1, 1)
    figure.append_trace(line_roc, 1, 2)
    figure.append_trace(line_diagonal, 1, 2)
    
    figure['layout'].update(margin = {'l': 40})
    
    plotly.offline.iplot(figure, show_link = False)


def model_group_monitor(in_data, target_variable, probability_var, higher_better = True, number_of_groups = 10):
    in_data = in_data[in_data[target_variable].notnull()]
    in_data = in_data.copy().reset_index().drop('index', axis = 1)
    
    try:
        bins = pd.DataFrame(pd.qcut(in_data[probability_var], number_of_groups, labels = False, 
                                    duplicates = 'drop')).rename(columns = {probability_var: 'bin_group'})
    except:
        try:
            bins = pd.DataFrame(pd.qcut(in_data[probability_var], number_of_groups, labels = False)).rename(columns = {probability_var: 'bin_group'})
        except:
            bins = pd.DataFrame(pd.cut(in_data[probability_var], number_of_groups, labels = False)).rename(columns = {probability_var: 'bin_group'})
    
    if higher_better == False:
        bins['adverse_bin_group'] = bins['bin_group'].apply(lambda x: number_of_groups - 1 - x)
        bins.drop('bin_group', axis = 1, inplace = True)
        bins.rename(columns = {'adverse_bin_group': 'bin_group'}, inplace = True)
    
    in_data = in_data.merge(bins, how = 'left', left_index = True, right_index = True)
    
    in_data['cnt_group'] = 1
    in_data['cnt_bad'] = in_data[target_variable].apply(lambda x: 1 if x == 1 else 0)
    in_data['cnt_good'] = in_data[target_variable].apply(lambda x: 1 if x == 0 else 0)
    in_data[probability_var + '_min'] = in_data[probability_var]
    in_data[probability_var + '_max'] = in_data[probability_var]
    in_data[probability_var + '_avg'] = in_data[probability_var]
    
    in_data_grouped = in_data.groupby('bin_group').agg({'cnt_group': np.sum, 'cnt_bad': np.sum, 'cnt_good': np.sum,
        probability_var + '_min': np.min, probability_var + '_max': np.max, probability_var + '_avg': np.average})
    in_data_grouped['group_bad_rate'] = in_data_grouped['cnt_bad'] / in_data_grouped['cnt_group']
    
    in_data_grouped['total_bad'] = np.sum(in_data_grouped['cnt_bad'])
    in_data_grouped['total_good'] = np.sum(in_data_grouped['cnt_good'])
    in_data_grouped['total_bad_rate'] = in_data_grouped['total_bad'] / (in_data_grouped['total_bad'] + in_data_grouped['total_good'])
    
    in_data_grouped['cum_bad'] = in_data_grouped['cnt_bad'].cumsum()
    in_data_grouped['cum_good'] = in_data_grouped['cnt_good'].cumsum()
    
    in_data_grouped['cum_bad_pct'] = in_data_grouped['cum_bad'] / in_data_grouped['total_bad']
    in_data_grouped['cum_good_pct'] = in_data_grouped['cum_good'] / in_data_grouped['total_good']
    in_data_grouped['KS'] = in_data_grouped['cum_bad_pct'] - in_data_grouped['cum_good_pct']
    in_data_grouped['KS'] = in_data_grouped['KS'].apply(lambda x: abs(x))
    
    in_data_grouped = in_data_grouped.reset_index()
    in_data_grouped['bin_group'] = in_data_grouped['bin_group'].apply(lambda x: x + 1)
    
    in_data_grouped = in_data_grouped[['total_bad', 'total_good', 'total_bad_rate', 'cnt_group', 'cnt_bad', 
                                       'cnt_good', 'group_bad_rate', 'cum_bad', 'cum_good', 'cum_bad_pct', 'cum_good_pct', 'KS', 
                                       probability_var + '_min', probability_var + '_max', probability_var + '_avg']]
    
    return in_data_grouped


def variable_woe(in_data, target_variable, x_variable):
    in_data = in_data.copy()
    
    in_data['cnt_bad'] = in_data[target_variable].apply(lambda x: 1 if x == 1 else 0)
    in_data['cnt_good'] = in_data[target_variable].apply(lambda x: 1 if x == 0 else 0)
    
    in_data_grouped = in_data.groupby(x_variable).agg({'cnt_bad': np.sum, 'cnt_good': np.sum})
    
    in_data_grouped['bad_rate'] = in_data_grouped['cnt_bad'] / (in_data_grouped['cnt_bad'] + in_data_grouped['cnt_good'])
    
    in_data_grouped['cnt_bad_total'] = np.sum(in_data_grouped['cnt_bad'])
    in_data_grouped['cnt_good_total'] = np.sum(in_data_grouped['cnt_good'])
    
    in_data_grouped['pct_bad'] = in_data_grouped['cnt_bad'] / in_data_grouped['cnt_bad_total']
    in_data_grouped['pct_good'] = in_data_grouped['cnt_good'] / in_data_grouped['cnt_good_total']
    
    in_data_grouped['WoE'] = in_data_grouped[['pct_good', 'pct_bad']].apply(lambda x: 
        10e8 if x['pct_bad'] == 0 else -10e8 if x['pct_good'] == 0 else np.log(x['pct_good'] / x['pct_bad']) * 100, axis = 1)
    
    in_data_grouped = in_data_grouped.reset_index().rename(columns = {x_variable: 'value'})
    in_data_grouped['variable'] = x_variable
    
    in_data_grouped = in_data_grouped[['variable', 'value', 'cnt_bad', 'cnt_good', 'bad_rate', 'cnt_bad_total', 
                                       'cnt_good_total', 'pct_bad', 'pct_good', 'WoE']]
    
    return in_data_grouped


def score_distribution(in_data, probability_var, number_of_groups = 10):
    in_data = in_data.copy().reset_index().drop('index', axis = 1)
    
    bins = pd.DataFrame(pd.qcut(in_data[[probability_var]], number_of_groups, labels = False)).rename(columns = {0: 'bin_group'})
    in_data = in_data.merge(bins, how = 'left', left_index = True, right_index = True)
    
    in_data['cnt_group'] = 1
    in_data[probability_var + '_min'] = in_data[probability_var]
    in_data[probability_var + '_max'] = in_data[probability_var]
    in_data[probability_var + '_avg'] = in_data[probability_var]
    
    in_data_grouped = in_data.groupby('bin_group').agg({'cnt_group': np.sum, 
        probability_var + '_min': np.min, probability_var + '_max': np.max, probability_var + '_avg': np.average})
    
    in_data_grouped = in_data_grouped.reset_index()
    in_data_grouped['bin_group'] = in_data_grouped['bin_group'].apply(lambda x: x + 1)
    
    in_data_grouped = in_data_grouped[['bin_group', 'cnt_group', 
                                       probability_var + '_min', probability_var + '_max', probability_var + '_avg']]
    
    return in_data_grouped


def variable_distribution(in_data, x_variable):
    in_data = in_data.copy()
    
    in_data['group_total'] = 1
    
    in_data_grouped = in_data.groupby(x_variable).agg({'group_total': np.sum})
    in_data_grouped['cnt_total'] = np.sum(in_data_grouped['group_total'])
    in_data_grouped['group_pct'] = in_data_grouped['group_total'] / in_data_grouped['cnt_total']
    
    in_data_grouped = in_data_grouped.reset_index().rename(columns = {x_variable: 'value'})
    in_data_grouped['variable'] = x_variable
    
    in_data_grouped = in_data_grouped[['variable', 'value', 'group_total', 'cnt_total', 'group_pct']]
    
    return in_data_grouped


def ks_variation_over_time(in_data, y_variable, probability_var, score_var, time_variable, frequency = 'WEEKLY'):
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    unique_time = pd.Series(in_data['time_interval'].unique()).sort_values()
    ks_list = list()
    ks_list1= list()
    auc_list = list()
    quantity_list = list()
    bad_rate_list = list()
    
    for i in unique_time:
        sample = in_data[in_data['time_interval'] == i].copy()
        sample_data = in_data[in_data['time_interval'] == i].copy()
        quantity = sample.shape[0]
        sample['1_y'] = 1 - sample[y_variable]
        sample_grouped = sample[[probability_var, y_variable, '1_y']].groupby(probability_var).agg({y_variable: np.sum, '1_y': np.sum}).reset_index()
        sample_grouped['cum_pct_0'] = sample_grouped['1_y'].cumsum() / (sample.shape[0] - np.sum(sample[y_variable]))
        sample_grouped['cum_pct_1'] = sample_grouped[y_variable].cumsum() / np.sum(sample[y_variable])
        sample_grouped['ks'] = (sample_grouped['cum_pct_0'] - sample_grouped['cum_pct_1']) * 100
        max_ks = sample_grouped.loc[sample_grouped['ks'].idxmax() : sample_grouped['ks'].idxmax()]
        
        sample_data['1_y'] = 1 - sample_data[y_variable]		
        model_data_grouped = sample_data[[score_var, y_variable, '1_y']].groupby(score_var).agg({
    																			y_variable: np.sum, '1_y': np.sum}).reset_index()
    																													
        model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (sample_data.shape[0] - np.sum(sample_data[y_variable]))
        model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(sample_data[y_variable])
        model_data_grouped['ks'] = abs((model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100)
        max_ks1 = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]
        
        quantity_list.append(quantity)
        if max_ks.size == 0:
            ks_score = ks_list[-1] if len(ks_list)>0 else 0
            ks_list.append(ks_score)
            ks_list1.append(ks_score)
            bad_rate_list.append(0)
        else:
            ks_list.append(int(max_ks['ks'].iloc[0]))
            ks_list1.append(int(max_ks1['ks'].iloc[0]))
            bad_rate_list.append(round(sample[y_variable].mean(), 3))
        
        try:
            auc_score = round(roc_auc_score(sample[y_variable], sample[probability_var]), 2)
            auc_list.append(auc_score)
        except:
            score = auc_list[-1] if len(auc_list)>0 else 0
            auc_list.append(score)
    
    #plt.figure(figsize = (10, 6))
    #plt.subplot(2, 1, 1)
    #plt.plot(range(len(ks_list)), ks_list, linewidth = 2)
    ##plt.xticks(range(len(ks_list)), unique_time, rotation='vertical')
    #plt.title('Model KS Variation')
    #plt.ylim(ymin = 0, ymax = 100)
    
    #plt.subplot(2, 1, 2)
    #plt.plot(range(len(auc_list)), auc_list, linewidth = 2)
    #plt.xticks(range(len(auc_list)), unique_time, rotation='vertical')
    #plt.title('Model Auc Variation')
    #plt.ylim(ymin = .5, ymax = 1)
    #plt.show()
    return unique_time.tolist(), auc_list, ks_list, ks_list1, quantity_list, bad_rate_list

def ks_and_score_variation_over_time_plotly(in_data, y_variable, probability_var, time_variable, frequency = 'WEEKLY'):
    import plotly
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    bad_rate_grouped = in_data.groupby('time_interval')[y_variable].agg({'cnt_total': np.size, 'cnt_bad': np.sum}).reset_index()
    bad_rate_grouped['bad_rate'] = bad_rate_grouped['cnt_bad'] / bad_rate_grouped['cnt_total']
    
    unique_time = pd.Series(in_data['time_interval'].unique()).sort_values()
    ks_list = list()
    
    for i in unique_time:
        sample = in_data[in_data['time_interval'] == i].copy()
        sample['1_y'] = 1 - sample[y_variable]
        sample_grouped = sample[[probability_var, y_variable, '1_y']].groupby(probability_var).agg({
            y_variable: np.sum, '1_y': np.sum}).reset_index()
        
        sample_grouped['cum_pct_0'] = sample_grouped['1_y'].cumsum() / (sample.shape[0] - np.sum(sample[y_variable]))
        sample_grouped['cum_pct_1'] = sample_grouped[y_variable].cumsum() / np.sum(sample[y_variable])
        sample_grouped['ks'] = abs((sample_grouped['cum_pct_1'] - sample_grouped['cum_pct_0']) * 100)
        max_ks = sample_grouped.loc[sample_grouped['ks'].idxmax() : sample_grouped['ks'].idxmax()]
        
        if max_ks.size == 0:
            ks_list.append(0)
        else:
            ks_list.append(max_ks['ks'].iloc[0])
    
    score_grouped = in_data.groupby('time_interval').agg({probability_var: np.average}).reset_index()
    
    line_ks = plotly.graph_objs.Scatter(name = 'KS', x = unique_time, y = ks_list, yaxis = 'y1')
    line_score = plotly.graph_objs.Scatter(name = 'Score', x = unique_time, y = score_grouped[probability_var], yaxis = 'y2')
    line_bad_rate = plotly.graph_objs.Scatter(
        name = 'Bad Rate', x = bad_rate_grouped['time_interval'], y = bad_rate_grouped['bad_rate'], yaxis = 'y3')
    
    figure = plotly.tools.make_subplots(rows = 1, cols = 3, print_grid = False, 
                                        subplot_titles = ['Model KS Variation', 'Model Score Variation', 'Bad Rate Variation'])
    
    figure.append_trace(line_ks, 1, 1)
    figure.append_trace(line_score, 1, 2)
    figure.append_trace(line_bad_rate, 1, 3)
    
    figure['layout']['yaxis1'].update(range = [0, 80])
    figure['layout']['yaxis2'].update(range = [np.min(score_grouped[probability_var]) - 20, 
                                               np.max(score_grouped[probability_var]) + 20])
    figure['layout']['yaxis3'].update(range = [0, np.max(bad_rate_grouped['bad_rate']) + 0.1])
    figure['layout'].update(height = 360, margin = {'l': 40})
    
    plotly.offline.iplot(figure, show_link = False)


def score_variation_over_time(in_data, probability_var, time_variable, frequency = 'WEEKLY'):
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    score_grouped = in_data.groupby('time_interval').agg({probability_var: np.average}).reset_index()
    
    plt.plot(range(score_grouped.index.size), score_grouped[probability_var], linewidth = 2)
    plt.xticks(range(score_grouped.index.size), score_grouped['time_interval'], rotation='vertical')
    plt.title('Model Score Variation')
    plt.ylabel('Information Value', size = 14)
    plt.show()


def variable_distribution_over_time(in_data, x_variable, time_variable, frequency = 'WEEKLY'):
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    time_by_var_grouped = in_data.groupby(['time_interval', x_variable], as_index = False)[x_variable].agg({'cnt': np.count_nonzero})
    time_by_var_grouped = time_by_var_grouped.pivot(index = 'time_interval', columns = x_variable, values = 'cnt').reset_index()
    
    time_grouped = in_data.groupby('time_interval', as_index = False)['time_interval'].agg({'cnt': np.count_nonzero})
    time_by_var_grouped = time_by_var_grouped.merge(time_grouped, how = 'left', on = 'time_interval').reset_index()
    time_by_var_grouped = time_by_var_grouped.fillna(0)
    
    unique_var = pd.Series(in_data[x_variable].unique()).sort_values()
    
    for i in unique_var:
        time_by_var_grouped[i + ' pct'] = time_by_var_grouped[i] / time_by_var_grouped['cnt']
        plt.plot(time_by_var_grouped['index'], time_by_var_grouped[i + ' pct'], linewidth = 2)
    
    plt.xticks(time_by_var_grouped['index'], time_by_var_grouped['time_interval'], rotation='vertical')
    plt.legend(bbox_to_anchor = [-0.1, 1])
    plt.title(x_variable + ' Distribution')
    plt.ylabel('Pct', size = 14)
    plt.show()
    
    return time_by_var_grouped


def variable_woe_over_time(in_data, x_variable, y_variable, time_variable, frequency = 'WEEKLY'):
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    unique_time = pd.Series(in_data['time_interval'].unique()).sort_values()
    woe_total = pd.DataFrame()
    
    for i in unique_time:
        sample = in_data[in_data['time_interval'] == i].copy()
        sample_out, woe_result = create_woe(sample, x_variable, y_variable)
        woe_result = woe_result[[x_variable + '_bin', x_variable + '_bin_woe_old']]
        woe_result['time_interval'] = i
        woe_total = woe_total.append(woe_result)
    
    woe_total = woe_total.pivot(
        index = 'time_interval', columns = x_variable + '_bin', values = x_variable + '_bin_woe_old').reset_index()
    woe_total = woe_total.replace(10e8, np.nan)
    woe_total = woe_total.replace(-10e8, np.nan)
    
    unique_var = pd.Series(in_data[x_variable].unique()).sort_values()
    
    for i in unique_var:
        plt.plot(range(woe_total.index.size), woe_total[i], linewidth = 2)
    
    plt.xticks(range(woe_total.index.size), woe_total['time_interval'], rotation='vertical')
    plt.legend(bbox_to_anchor = [-0.1, 1])
    plt.title(x_variable + ' WoE')
    plt.ylabel('Pct', size = 14)
    plt.show()


def variable_iv_over_time(in_data, x_variable, y_variable, time_variable, frequency = 'WEEKLY'):
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    unique_time = pd.Series(in_data['time_interval'].unique()).sort_values()
    woe_total = pd.DataFrame()
    
    for i in unique_time:
        try:
            sample = in_data[in_data['time_interval'] == i].copy()
            sample_out, woe_result = create_woe(sample, x_variable, y_variable)
            woe_result['variable'] = x_variable
            woe_result['time_interval'] = i
            woe_result = woe_result[['time_interval', 'variable', 'iv']].head(1)
            woe_total = woe_total.append(woe_result)
        except:
            pass
    
    plt.plot(range(woe_total.index.size), woe_total['iv'], linewidth = 2)
    plt.xticks(range(woe_total.index.size), woe_total['time_interval'], rotation='vertical')
    plt.title(x_variable + ' IV')
    plt.ylabel('Information Value', size = 14)
    plt.ylim(ymin = 0, ymax = 1)
    plt.show()


def variable_dist_woe_iv_over_time(in_data, x_variable, y_variable, time_variable, frequency = 'WEEKLY'):
    import plotly
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    time_by_var_grouped = in_data.groupby(['time_interval', x_variable], as_index = False)[x_variable].agg({'cnt': np.count_nonzero})
    time_by_var_grouped = time_by_var_grouped.pivot(index = 'time_interval', columns = x_variable, values = 'cnt').reset_index()
    
    time_grouped = in_data.groupby('time_interval', as_index = False)['time_interval'].agg({'cnt': np.count_nonzero})
    time_by_var_grouped = time_by_var_grouped.merge(time_grouped, how = 'left', on = 'time_interval').reset_index()
    time_by_var_grouped = time_by_var_grouped.fillna(0)
    
    unique_time = pd.Series(in_data['time_interval'].unique()).sort_values()
    woe_total = pd.DataFrame()
    iv_total = pd.DataFrame()
    
    for i in unique_time:
        sample = in_data[in_data['time_interval'] == i].copy()
        sample_out, woe_result = create_woe(sample, x_variable, y_variable)
        woe_result = woe_result[[x_variable + '_bin', x_variable + '_bin_woe_old', 'iv']]
        woe_result['time_interval'] = i
        woe_total = woe_total.append(woe_result)
        iv_total = iv_total.append(woe_result.head(1))
    
    woe_total = woe_total.pivot(
        index = 'time_interval', columns = x_variable + '_bin', values = x_variable + '_bin_woe_old').reset_index()
    woe_total = woe_total.replace(10e8, np.nan)
    woe_total = woe_total.replace(-10e8, np.nan)
    
    figure = plotly.tools.make_subplots(
        rows = 1, cols = 3, print_grid = False, subplot_titles = ['Distribution', 'WoE', 'Information Value'])
    
    unique_var = pd.Series(in_data[x_variable].unique()).sort_values()
    for i in unique_var:
        time_by_var_grouped[i] = time_by_var_grouped[i] / time_by_var_grouped['cnt']
        
        line_dist = plotly.graph_objs.Bar(
            name = i, x = time_by_var_grouped['time_interval'], y = time_by_var_grouped[i], legendgroup = 'g1')
        line_woe = plotly.graph_objs.Scatter(name = i, x = woe_total['time_interval'], y = woe_total[i], legendgroup = 'g2')
        
        figure.append_trace(line_dist, 1, 1)
        figure.append_trace(line_woe, 1, 2)
    
    line_iv = plotly.graph_objs.Scatter(name = 'IV', x = iv_total['time_interval'], y = iv_total['iv'], legendgroup = 'g3')
    figure.append_trace(line_iv, 1, 3)
    
    figure['layout'].update(height = 360, margin = {'l': 40}, title = x_variable, barmode = 'stack', legend = dict(orientation = 'v'))
    figure['layout']['yaxis3'].update(range = [0, 0.8])
    plotly.offline.iplot(figure, show_link = False)


def variable_distribution_over_time_plotly(in_data, x_variable, time_variable, frequency = 'WEEKLY'):
    import plotly
    in_data = in_data.copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    time_by_var_grouped = in_data.groupby(['time_interval', x_variable], as_index = False)[x_variable].agg({'cnt': np.count_nonzero})
    time_by_var_grouped = time_by_var_grouped.pivot(index = 'time_interval', columns = x_variable, values = 'cnt').reset_index()
    
    time_grouped = in_data.groupby('time_interval', as_index = False)['time_interval'].agg({'cnt': np.count_nonzero})
    time_by_var_grouped = time_by_var_grouped.merge(time_grouped, how = 'left', on = 'time_interval').reset_index()
    time_by_var_grouped = time_by_var_grouped.fillna(0)
    
    figure = plotly.tools.make_subplots(rows = 1, cols = 2, print_grid = False, subplot_titles = ['Distribution', 'Total Count'])
    
    unique_var = pd.Series(in_data[x_variable].unique()).sort_values()
    for i in unique_var:
        time_by_var_grouped[i + '_pct'] = time_by_var_grouped[i] / time_by_var_grouped['cnt']
        
        line_dist = plotly.graph_objs.Bar(
            name = i, x = time_by_var_grouped['time_interval'], y = time_by_var_grouped[i + '_pct'], legendgroup = 'g1')
        line_cnt = plotly.graph_objs.Scatter(
            name = i, x = time_by_var_grouped['time_interval'], y = time_by_var_grouped[i], legendgroup = 'g2')
        
        figure.append_trace(line_dist, 1, 1)
        figure.append_trace(line_cnt, 1, 2)
    
    figure['layout'].update(height = 400, margin = {'l': 40}, title = x_variable, barmode = 'stack', legend = dict(orientation = 'h'))
    plotly.offline.iplot(figure, show_link = False)


def logit_fit_all_models(sample_development, y_variable, x_list_or_dict, sample_validation = None, 
                         num_of_variables = 2, p_value_filter = 0.05):
    if type(x_list_or_dict) not in [list, dict]:
        raise ValueError('x_list_or_dict must be a list or dict')
    
    if len(x_list_or_dict) == 0:
        raise ValueError('x_list_or_dict must have at least one element')
    
    if num_of_variables <= 0:
        raise ValueError('num_of_combinations must be greater than 0')
    
    if num_of_variables > len(x_list_or_dict):
        raise ValueError('num_of_combinations must be less than or equal to the length of x_list_or_dict')
    
    if type(num_of_variables) != int:
        raise ValueError('num_of_combinations must be an integer')
    
    if type(x_list_or_dict) == list:
        x_combinations = list(itertools.combinations(x_list_or_dict, num_of_variables))
    else:
        x_combinations = list(itertools.combinations(list(x_list_or_dict.keys()), num_of_variables))
    
    num_total = len(x_combinations)
    num_iter = 1
    
    total_result = pd.DataFrame(columns = ['model_num', 'variable', 'coeff', 'p_value', 'ks_development', 'ks_validation'])
    
    model_dev = sample_development[list(x_list_or_dict.keys()) + [y_variable]].copy()
    model_val = sample_validation[ list(x_list_or_dict.keys()) + [y_variable]].copy()
    model_dev['const'] = 1
    model_val['const'] = 1
    
    time_zero = time.time()
    
    for i in x_combinations:
        if num_iter % 50 == 1:
            remaining_time = '%s sec.' % int(((time.time() - time_zero) / num_iter) * (num_total - num_iter))
            if num_iter + 49 > num_total:
                print('Total: %s. Fitting: %s to %s. Remaining time: %s' % (num_total, num_iter, num_total, remaining_time))
            else:
                print('Total: %s. Fitting: %s to %s. Remaining time: %s' % (num_total, num_iter, num_iter + 49, remaining_time))
        
        num_iter += 1
        
        x_list = list(i)
        
        logit_reg = sm.Logit(model_dev[y_variable], model_dev[['const'] + x_list])
        result = logit_reg.fit(disp = False)
        
        if (result.pvalues > p_value_filter).any():
            continue
        
        wrong_coeff = False
        if type(x_list_or_dict) == dict:
            for j in i:
                if (x_list_or_dict[j].lower() == 'negative' and result.params[j] >= 0) or \
                    (x_list_or_dict[j].lower() == 'positive' and result.params[j] <= 0):
                        wrong_coeff = True
                        break
        
        if wrong_coeff:
            continue
        
        model_dev['prob'] = result.predict()
        model_dev['1_y'] = 1 - model_dev[y_variable]
        model_data_grouped = model_dev.groupby('prob', as_index = False).agg({y_variable: np.sum, '1_y': np.sum})
        model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (model_dev.shape[0] - model_dev[y_variable].sum())
        model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / model_dev[y_variable].sum()
        max_ks_dev = (model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']).max() * 100
        
        if sample_validation is not None:
            logit_reg = sm.Logit(model_val[y_variable], model_val[['const'] + x_list])
            model_val['prob'] = logit_reg.predict(result.params)
            
            model_val['1_y'] = 1 - model_val[y_variable]
            model_data_grouped = model_val.groupby('prob', as_index = False).agg({y_variable: np.sum, '1_y': np.sum})
            model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (model_val.shape[0] - model_val[y_variable].sum())
            model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / model_val[y_variable].sum()
            max_ks_val = (model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']).max() * 100
        else:
            max_ks_val = np.nan
        
        single = pd.DataFrame({'model_num': num_iter, 'variable': result.params.index, 'coeff': result.params.values, 
                               'p_value': result.pvalues.values, 'ks_development': max_ks_dev, 'ks_validation': max_ks_val}, 
                               columns = ['model_num', 'variable', 'coeff', 'p_value', 'ks_development', 'ks_validation'], 
                               index = range(result.params.index.size))
        total_result = pd.concat([total_result, single])
    
    total_result = total_result.reset_index().rename(columns = {'index': 'variable_num'})
    total_result = total_result[['model_num', 'variable_num', 'variable', 'coeff', 'p_value', 'ks_development', 'ks_validation']]
    total_result['model_num'] = total_result['model_num'].astype(np.int)
    total_result = total_result.sort_values(['ks_development', 'model_num', 'variable_num'], ascending = [False, True, True])
    
    return total_result


def fit_listed_models(model_input):
    sample_development = model_input[0]
    y_variable = model_input[1]
    x_combinations = model_input[2]
    sample_validation = model_input[3]
    
    num_total = len(x_combinations)
    num_iter = 1
    time_zero = time.time()
    
    total_result = pd.DataFrame(columns = ['model_num', 'variable', 'coeff', 'p_value', 'ks_development', 'ks_validation'])
    
    for i in x_combinations:
        if num_iter % 20 == 1:
            remaining_time = 'estimating'
            
            if num_iter > 20:
                remaining_time = ((time.time() - time_zero) / num_iter) * (num_total - num_iter)
                
                if remaining_time > 3600:
                    remaining_time = str(int(remaining_time / 3600)) + ' hours ' + str(int(remaining_time % 3600)) + ' min.'
                elif remaining_time > 60:
                    remaining_time = str(int(remaining_time / 60)) + ' min ' + str(int(remaining_time % 60)) + ' sec.'
                else:
                    remaining_time = str(int(remaining_time)) + ' sec.'
            
            if num_iter + 19 > num_total:
                print('Total: ' + str(num_total) + '. Fitting: ' + str(num_iter) + ' to ' 
                      + str(num_total) + '. Remaining time: ' + remaining_time)
            else:
                print('Total: ' + str(num_total) + '. Fitting: ' + str(num_iter) + ' to ' 
                      + str(num_iter + 19) + '. Remaining time: ' + remaining_time)
        
        num_iter = num_iter + 1
        
        x_list = list(i)
        
        model_dev = sm.add_constant(sample_development[x_list + [y_variable]])
        logit_reg = sm.Logit(sample_development[y_variable], model_dev[['const'] + x_list])
        result = logit_reg.fit(disp = False)
        
        model_dev['prob'] = result.predict()
        
        model_dev['1_y'] = 1 - model_dev[y_variable]
        model_data_grouped = model_dev[['prob', y_variable, '1_y']].groupby('prob').agg(
            {y_variable: np.sum, '1_y': np.sum}).reset_index()
        model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (
        model_dev.shape[0] - np.sum(model_dev[y_variable]))
        model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(model_dev[y_variable])
        model_data_grouped['ks'] = (model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100
        max_ks_dev = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]['ks'].iloc[0]
        
        if sample_validation is not None:
            model_val = sm.add_constant(sample_validation[x_list + [y_variable]])
            logit_reg = sm.Logit(model_val['const'], model_val[['const'] + x_list])
            model_val['prob'] = logit_reg.predict(result.params)
            
            model_val['1_y'] = 1 - model_val[y_variable]
            model_data_grouped = model_val[['prob', y_variable, '1_y']].groupby('prob').agg(
                {y_variable: np.sum, '1_y': np.sum}).reset_index()
            model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (
            model_val.shape[0] - np.sum(model_val[y_variable]))
            model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(model_val[y_variable])
            model_data_grouped['ks'] = (model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100
            max_ks_val = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]['ks'].iloc[0]
        else:
            max_ks_val = np.nan
        
        single = pd.DataFrame(columns = ['model_num', 'variable', 'coeff', 'p_value', 'ks_development', 'ks_validation'])
        
        variable_coeffs = pd.DataFrame(result.params).reset_index().rename(columns = {'index': 'variable', 0: 'coeff'})
        variable_pvalues = pd.DataFrame(result.pvalues).reset_index().rename(columns = {'index': 'variable', 0: 'p_value'})
        coeffs_pvalues = variable_coeffs.merge(variable_pvalues, how = 'left', on = 'variable')
        
        single = single.append(coeffs_pvalues)
        single['model_num'] = num_iter
        single['ks_development'] = max_ks_dev
        single['ks_validation'] = max_ks_val
        total_result = total_result.append(single)
    
    return total_result


def logit_fit_all_models_multiprocess(sample_development, y_variable, x_list_or_dict, sample_validation = None, 
                         num_of_variables = 2, p_value_filter = 0.05, num_of_processes = None):
    if type(x_list_or_dict) not in [list, dict]:
        raise ValueError('x_list_or_dict must be a list or dict')
    
    if len(x_list_or_dict) == 0:
        raise ValueError('x_list_or_dict must have at least one element')
    
    if num_of_variables <= 0:
        raise ValueError('num_of_combinations must be greater than 0')
    
    if num_of_variables > len(x_list_or_dict):
        raise ValueError('num_of_combinations must be less than or equal to the length of x_list_or_dict')
    
    if type(num_of_variables) != int:
        raise ValueError('num_of_combinations must be an integer')
    
    if type(x_list_or_dict) == list:
        x_combinations = list(itertools.combinations(x_list_or_dict, num_of_variables))
    else:
        x_combinations = list(itertools.combinations(list(x_list_or_dict.keys()), num_of_variables))
    
    if num_of_processes is None:
        num_of_processes = os.cpu_count()
    
    x_chunks = []
    chunk_size = math.ceil(len(x_combinations) / num_of_processes)
    
    for i in range(0, len(x_combinations), chunk_size):
        x_chunks.append((sample_development, y_variable, x_combinations[i : i + chunk_size], sample_validation))
    
    with Pool(num_of_processes) as p:
        process_result = p.map(fit_listed_models, x_chunks)
    
    total_result = pd.DataFrame()
    
    for i in process_result:
        total_result = total_result.append(i)
    
    total_result = total_result.reset_index().rename(columns = {'index': 'variable_num'})
    total_result = total_result[['model_num', 'variable_num', 'variable', 'coeff', 'p_value', 'ks_development', 'ks_validation']]
    total_result['model_num'] = total_result['model_num'].astype(np.int)
    total_result = total_result.sort_values(['ks_development', 'model_num', 'variable_num'], ascending = [False, True, True])
    
    return total_result


def calculate_vif(in_data, x_variables):
    vif_df = pd.DataFrame(columns = ['variable', 'VIF'])
    vif_num = 0
    
    for i in x_variables:
        single_vif = pd.DataFrame({'variable': i, 
                          'VIF': sm_vif.variance_inflation_factor(np.array(in_data[x_variables]), vif_num)}, index = range(1, 2))
        vif_df = vif_df.append(single_vif)
        vif_num = vif_num + 1
    
    vif_df = vif_df[['variable', 'VIF']].reset_index().drop('index', axis = 1)
    
    return vif_df


def get_ks_roc_chart_data(in_data, y_variable, score_var, higher_better = True, score_interval = np.nan):
    model_data = in_data[in_data[y_variable].notnull()][[y_variable, score_var]].copy()
    
    if not np.isnan(score_interval):
        model_data[score_var] = model_data[score_var].apply(lambda x: math.ceil(x / score_interval) * score_interval)
    
    model_data['1_y'] = (1 - model_data[y_variable]).astype(np.float64)
    model_data_grouped = model_data[[score_var, y_variable, '1_y']].groupby(score_var).agg({
        y_variable: np.sum, '1_y': np.sum}).reset_index()
    
    sensitivity = model_data_grouped.sort_values(score_var, ascending = higher_better)
    sensitivity['cum_sum'] = sensitivity[y_variable].cumsum()
    sensitivity = sensitivity.sort_values('cum_sum', ascending = higher_better)['cum_sum'] / np.sum(model_data[y_variable])
    
    specificity = model_data_grouped.sort_values(score_var, ascending = (not higher_better))
    specificity['cum_sum'] = specificity['1_y'].cumsum()
    specificity = specificity.sort_values(
        'cum_sum', ascending = (not higher_better))['cum_sum'] / (model_data.shape[0] - np.sum(model_data[y_variable]))
    
    auc = abs((specificity.diff() * (sensitivity + sensitivity.shift().fillna(1))).sum() / 2)
    
    model_data_grouped['cum_pct_0'] = model_data_grouped['1_y'].cumsum() / (model_data.shape[0] - np.sum(model_data[y_variable]))
    model_data_grouped['cum_pct_1'] = model_data_grouped[y_variable].cumsum() / np.sum(model_data[y_variable])
    model_data_grouped['ks'] = abs((model_data_grouped['cum_pct_0'] - model_data_grouped['cum_pct_1']) * 100)
    max_ks = model_data_grouped.loc[model_data_grouped['ks'].idxmax() : model_data_grouped['ks'].idxmax()]
    
    roc_x = (1 - specificity).rename('one_minus_specificity')
    roc_y = sensitivity.rename('sensitivity')
    
    del model_data, model_data_grouped[y_variable], model_data_grouped['1_y']
    
    output_dict = {}
    output_dict['ks_data'] = model_data_grouped
    output_dict['roc_data'] = pd.concat([roc_x, roc_y], axis = 1)
    output_dict['max_ks'] = max_ks['ks'].iloc[0]
    output_dict['max_ks_score'] = max_ks[score_var].iloc[0]
    output_dict['max_ks_cum_pct_0'] = max_ks['cum_pct_0'].iloc[0]
    output_dict['max_ks_cum_pct_1'] = max_ks['cum_pct_1'].iloc[0]
    output_dict['auc'] = auc
    
    return output_dict


def get_ks_roc_chart_json(in_data, y_variable, score_var, higher_better = True, score_interval = np.nan, time_var = None):
    in_data = in_data[in_data[y_variable].notnull()][[y_variable, score_var, time_var]].copy()
    ks_and_roc = get_ks_roc_chart_data(in_data, y_variable = y_variable, score_var = score_var, 
                                       higher_better = higher_better, score_interval = score_interval)
    
    ks_axis_y_0 = []
    for i in zip(ks_and_roc['ks_data'][score_var], ks_and_roc['ks_data']['cum_pct_0']):
        ks_axis_y_0.append([int(list(i)[0]), float(round(list(i)[1], 4))])
    
    ks_axis_y_1 = []
    for i in zip(ks_and_roc['ks_data'][score_var], ks_and_roc['ks_data']['cum_pct_1']):
        ks_axis_y_1.append([int(list(i)[0]), float(round(list(i)[1], 4))])
    
    roc = []
    for i in zip(ks_and_roc['roc_data']['one_minus_specificity'], ks_and_roc['roc_data']['sensitivity']):
        roc.append([float(round(list(i)[0], 4)), float(round(list(i)[1], 4))])
    
    ks_roc_json = {}
    ks_roc_json['max_ks'] = float(round(ks_and_roc['max_ks'], 4))
    ks_roc_json['max_ks_score'] = int(ks_and_roc['max_ks_score'])
    ks_roc_json['max_ks_cum_pct_0'] = float(round(ks_and_roc['max_ks_cum_pct_0'], 4))
    ks_roc_json['max_ks_cum_pct_1'] = float(round(ks_and_roc['max_ks_cum_pct_1'], 4))
    ks_roc_json['auc'] = float(round(ks_and_roc['auc'], 4))
    ks_roc_json['ks_axis_y_0'] = ks_axis_y_0
    ks_roc_json['ks_axis_y_1'] = ks_axis_y_1
    ks_roc_json['roc'] = roc
    ks_roc_json['target_variable'] = y_variable
    
    if time_var != None:
        ks_roc_json['create_date_min'] = np.min(in_data[time_var]).strftime('%Y-%m-%d')
        ks_roc_json['create_date_max'] = np.max(in_data[time_var]).strftime('%Y-%m-%d')
    else:
        ks_roc_json['create_date_min'] = 'N/A time'
        ks_roc_json['create_date_max'] = 'N/A time'
    
    return ks_roc_json


def ks_score_over_time_json(in_data, y_variable, score_var, time_variable, frequency = 'WEEKLY'):
    in_data = in_data[in_data[y_variable].notnull()][[y_variable, score_var, time_variable]].copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    cnt_total = in_data.groupby('time_interval', as_index = False).agg({y_variable: np.size})
    cnt_bad = in_data.groupby('time_interval', as_index = False).agg({y_variable: np.sum})
    bad_rate = cnt_bad[y_variable] / cnt_total[y_variable]
    
    unique_time = pd.Series(in_data['time_interval'].unique()).sort_values()
    ks_list = list()
    
    in_data['1_y'] = 1 - in_data[y_variable]
    
    for i in unique_time:
        sample = in_data[in_data['time_interval'] == i]
        sample_grouped = sample[[score_var, y_variable, '1_y']].groupby(score_var, as_index = False).agg({
            y_variable: np.sum, '1_y': np.sum})
        
        sample_grouped['cum_pct_0'] = sample_grouped['1_y'].cumsum() / (sample.shape[0] - np.sum(sample[y_variable]))
        sample_grouped['cum_pct_1'] = sample_grouped[y_variable].cumsum() / np.sum(sample[y_variable])
        sample_grouped['ks'] = abs((sample_grouped['cum_pct_1'] - sample_grouped['cum_pct_0']) * 100)
        max_ks = sample_grouped.loc[sample_grouped['ks'].idxmax() : sample_grouped['ks'].idxmax()]
        
        if max_ks.size == 0:
            ks_list.append(0)
        else:
            ks_list.append(max_ks['ks'].iloc[0])
    
    score_grouped = in_data.groupby('time_interval', as_index = False).agg({score_var: np.average})
    
    ks_over_time_dict = {}
    ks_over_time_dict['axis_time'] = list(unique_time)
    ks_over_time_dict['ks_series'] = [float(round(x, 4)) for x in ks_list]
    ks_over_time_dict['score_series'] = list(float(round(x, 4)) for x in score_grouped[score_var])
    ks_over_time_dict['bad_rate_series'] = list(float(round(x, 4)) for x in bad_rate)
    ks_over_time_dict['observations'] = list(int(x) for x in cnt_total[y_variable])
    
    return ks_over_time_dict


def model_group_monitor_json(in_data, target_variable, probability_var, higher_better = True, number_of_groups = 10):
    in_data = in_data[in_data[target_variable].notnull()][[target_variable, probability_var]].copy()
    in_data = in_data.reset_index().drop('index', axis = 1)
    
    try:
        bins = pd.DataFrame(pd.qcut(in_data[[probability_var]], number_of_groups, labels = False)).rename(columns = {0: 'bin_group'})
    except:
        bins = pd.DataFrame(pd.qcut(in_data[[probability_var]], 5, labels = False)).rename(columns = {0: 'bin_group'})
    
    if higher_better == False:
        bins['adverse_bin_group'] = bins['bin_group'].apply(lambda x: number_of_groups - 1 - x)
        bins.drop('bin_group', axis = 1, inplace = True)
        bins.rename(columns = {'adverse_bin_group': 'bin_group'}, inplace = True)
    
    in_data = in_data.merge(bins, how = 'left', left_index = True, right_index = True)
    
    in_data['cnt_group'] = 1
    
    in_data['cnt_bad'] = np.where(in_data[target_variable] == 1, 1, 0)
    in_data['cnt_good'] = np.where(in_data[target_variable] == 0, 1, 0)
    in_data[probability_var + '_min'] = in_data[probability_var]
    in_data[probability_var + '_max'] = in_data[probability_var]
    in_data[probability_var + '_avg'] = in_data[probability_var]
    
    in_data_grouped = in_data.groupby('bin_group').agg({'cnt_group': np.sum, 'cnt_bad': np.sum, 'cnt_good': np.sum,
        probability_var + '_min': np.min, probability_var + '_max': np.max, probability_var + '_avg': np.average})
    in_data_grouped['group_bad_rate'] = in_data_grouped['cnt_bad'] / in_data_grouped['cnt_group']
    
    in_data_grouped['total_bad'] = np.sum(in_data_grouped['cnt_bad'])
    in_data_grouped['total_good'] = np.sum(in_data_grouped['cnt_good'])
    in_data_grouped['total_bad_rate'] = in_data_grouped['total_bad'] / (in_data_grouped['total_bad'] + in_data_grouped['total_good'])
    
    in_data_grouped['cum_bad'] = in_data_grouped['cnt_bad'].cumsum()
    in_data_grouped['cum_good'] = in_data_grouped['cnt_good'].cumsum()
    
    in_data_grouped['cum_bad_pct'] = in_data_grouped['cum_bad'] / in_data_grouped['total_bad']
    in_data_grouped['cum_good_pct'] = in_data_grouped['cum_good'] / in_data_grouped['total_good']
    in_data_grouped['KS'] = in_data_grouped['cum_bad_pct'] - in_data_grouped['cum_good_pct']
    in_data_grouped['KS'] = in_data_grouped['KS'].apply(lambda x: abs(x))
    
    in_data_grouped = in_data_grouped.reset_index()
    in_data_grouped['bin_group'] = in_data_grouped['bin_group'].apply(lambda x: x + 1)
    
    in_data_grouped = in_data_grouped[['total_bad', 'total_good', 'total_bad_rate', 'cnt_group', 'cnt_bad', 
                                       'cnt_good', 'group_bad_rate', 'cum_bad', 'cum_good', 'cum_bad_pct', 'cum_good_pct', 'KS', 
                                       probability_var + '_min', probability_var + '_max', probability_var + '_avg']]
    
    model_group_dict = {}
    model_group_dict['total_cnt'] = int(in_data_grouped.iloc[0]['total_good'] + in_data_grouped.iloc[0]['total_bad'])
    model_group_dict['total_good'] = int(in_data_grouped.iloc[0]['total_good'])
    model_group_dict['total_bad'] = int(in_data_grouped.iloc[0]['total_bad'])
    model_group_dict['total_bad_rate'] = float(round(model_group_dict['total_bad'] / model_group_dict['total_cnt'], 4))
    model_group_dict['group_cnt'] = list(int(x) for x in in_data_grouped['cnt_group'])
    model_group_dict['group_bad'] = list(int(x) for x in in_data_grouped['cnt_bad'])
    model_group_dict['group_good'] = list(int(x) for x in in_data_grouped['cnt_good'])
    model_group_dict['group_bad_rate'] = list(float(round(x, 4)) for x in in_data_grouped['group_bad_rate'])
    model_group_dict['cum_bad'] = list(int(x) for x in in_data_grouped['cum_bad'])
    model_group_dict['cum_good'] = list(int(x) for x in in_data_grouped['cum_good'])
    model_group_dict['cum_bad_pct'] = list(float(round(x, 4)) for x in in_data_grouped['cum_bad_pct'])
    model_group_dict['cum_good_pct'] = list(float(round(x, 4)) for x in in_data_grouped['cum_good_pct'])
    model_group_dict['KS'] = list(float(round(x, 4)) for x in in_data_grouped['KS'])
    model_group_dict['score_min'] = list(float(round(x, 4)) for x in in_data_grouped[probability_var + '_min'])
    model_group_dict['score_max'] = list(float(round(x, 4)) for x in in_data_grouped[probability_var + '_max'])
    model_group_dict['score_avg'] = list(float(round(x, 4)) for x in in_data_grouped[probability_var + '_avg'])
    
    return model_group_dict


def get_distribution_woe_iv(in_data, bin_variable, target_variable):
    in_data = in_data[in_data[target_variable].notnull()][[bin_variable, target_variable]].copy()
    in_data['cnt'] = 1
    in_data['cnt_bad'] = in_data[target_variable].apply(lambda x: 1 if x == 1 else 0)
    in_data['cnt_good'] = in_data[target_variable].apply(lambda x: 0 if x == 1 else 1)
    
    grouped = in_data.groupby(bin_variable, as_index = False).agg({'cnt': np.sum, 'cnt_bad': np.sum, 'cnt_good': np.sum})
    grouped['total_cnt'] = np.sum(grouped['cnt'])
    grouped['pct_cnt'] = grouped['cnt'] / np.sum(grouped['cnt'])
    grouped['pct_bad'] = grouped['cnt_bad'] / np.sum(grouped['cnt_bad'])
    grouped['pct_good'] = grouped['cnt_good'] / np.sum(grouped['cnt_good'])
    
    grouped['good_to_bad_ratio'] = grouped['pct_good'] / grouped['pct_bad']
    grouped['woe'] = grouped['good_to_bad_ratio'].apply(lambda x: 
        np.nan if np.log(x) > 10e8 else np.nan if np.log(x) < -10e8 else np.log(x) * 100)
    grouped['iv'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe'] / 100
    grouped['iv'] = grouped['iv'].sum()
    
    return grouped


def variable_woe_over_time_json(in_data, bin_variable_list, target_variable, time_variable, frequency = 'WEEKLY'):
    in_data = in_data[in_data[target_variable].notnull()][bin_variable_list + [target_variable, time_variable]].copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    in_data['cnt'] = 1
    in_data['cnt_bad'] = np.where(in_data[target_variable] == 1, 1, 0)
    in_data['cnt_good'] = np.where(in_data[target_variable] == 1, 0, 1)
    
    bin_var_woe_list = []
    
    for single_var in bin_variable_list:
        grouped = in_data.groupby(['time_interval', single_var], as_index = False).agg({
            'cnt': np.sum, 'cnt_bad': np.sum, 'cnt_good': np.sum})
        
        grouped_cnt_total = grouped.groupby('time_interval', as_index = False).agg(
            {'cnt': np.sum, 'cnt_bad': np.sum, 'cnt_good': np.sum})
        grouped_cnt_total.rename(
            columns = {'cnt': 'total_cnt', 'cnt_bad': 'total_cnt_bad', 'cnt_good': 'total_cnt_good'}, inplace = True)
        grouped = grouped.merge(grouped_cnt_total, how = 'left', on = 'time_interval')
        
        grouped['bin_dist_pct'] = grouped['cnt'] / grouped['total_cnt']
        
        grouped['pct_cnt'] = grouped['cnt'] / grouped['total_cnt']
        grouped['pct_bad'] = grouped['cnt_bad'] / grouped['total_cnt_bad']
        grouped['pct_good'] = grouped['cnt_good'] / grouped['total_cnt_good']
        
        grouped['good_to_bad_ratio'] = grouped['pct_good'] / grouped['pct_bad']
        
        grouped['woe'] = grouped['good_to_bad_ratio'].apply(lambda x: 
            np.nan if np.log(x) > 10e8 else np.nan if np.log(x) < -10e8 else np.log(x) * 100)
        
        grouped['iv'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe'] / 100
        
        grouped_iv = grouped.groupby('time_interval', as_index = False).agg({'iv': np.sum})
        grouped.drop('iv', axis = 1, inplace = True)
        grouped = grouped.merge(grouped_iv, how = 'left', on = 'time_interval')
        
        unique_time = pd.Series(grouped['time_interval'].unique()).sort_values()
        unique_bin_value = pd.Series(grouped[single_var].unique()).sort_values()
        
        woe_over_time_dict = {}
        woe_over_time_dict['bin_variable'] = single_var
        woe_over_time_dict['bin_woe_series'] = {}
        woe_over_time_dict['bin_dist_series'] = {}
        woe_over_time_dict['iv_series'] = []
        
        for i in unique_bin_value:
            woe_over_time_dict['bin_woe_series'][i] = []
            woe_over_time_dict['bin_dist_series'][i] = []
        
        for i in unique_time:
            woe_df = grouped[grouped['time_interval'] == i]
            woe_over_time_dict['iv_series'].append(None if np.isnan(woe_df.iloc[0]['iv']) else float(round(woe_df.iloc[0]['iv'], 4)))
            
            for j in unique_bin_value:
                if (woe_df[woe_df[single_var] == j].index.size > 0):
                    if np.isnan(woe_df[woe_df[single_var] == j].iloc[0]['woe']):
                        woe_over_time_dict['bin_woe_series'][j].append(None)
                    else:
                        woe_over_time_dict['bin_woe_series'][j].append(
                            float(round(woe_df[woe_df[single_var] == j].iloc[0]['woe'], 4)))
                else:
                    woe_over_time_dict['bin_woe_series'][j].append(None)
            
            for j in unique_bin_value:
                if (woe_df[woe_df[single_var] == j].index.size > 0):
                    if np.isnan(woe_df[woe_df[single_var] == j].iloc[0]['pct_cnt']):
                        woe_over_time_dict['bin_dist_series'][j].append(None)
                    else:
                        woe_over_time_dict['bin_dist_series'][j].append(
                            float(round(woe_df[woe_df[single_var] == j].iloc[0]['pct_cnt'], 4)))
                else:
                    woe_over_time_dict['bin_dist_series'][j].append(None)
        
        bin_var_woe_list.append(woe_over_time_dict)
    
    return bin_var_woe_list


def get_model_tracker_json(in_data, x_var_list, overview_y_var, detail_y_var_list, score_var, time_var):
    chart_data_list = {}
    
    if type(detail_y_var_list) == str:
        detail_y_var_list = [detail_y_var_list]
    
    if type(x_var_list) == str:
        x_var_list = [x_var_list]
    
    for i in detail_y_var_list:
        single_overview_json = get_ks_roc_chart_json(in_data, y_variable = i, score_var = score_var, 
                                                     higher_better = True, score_interval = 10, time_var = time_var)
        single_ks_weekly = ks_score_over_time_json(in_data, i, score_var, time_var, 'weekly')
        single_ks_monthly = ks_score_over_time_json(in_data, i, score_var, time_var, 'monthly')
        
        single_model_groups = model_group_monitor_json(in_data, i, score_var)
        
        single_chart_data = {}
        single_chart_data['target_variable'] = i
        single_chart_data['overview'] = single_overview_json
        single_chart_data['ks_frequency'] = {}
        single_chart_data['ks_frequency']['weekly'] = single_ks_weekly
        single_chart_data['ks_frequency']['monthly'] = single_ks_monthly
        single_chart_data['model_groups'] = single_model_groups
        single_chart_data['monitor_frequency'] = {}
        single_chart_data['monitor_frequency']['weekly'] = []
        single_chart_data['monitor_frequency']['monthly'] = []
        
        var_woe_weekly = variable_woe_over_time_json(in_data, x_var_list, i, time_var, 'weekly')
        var_woe_monthly = variable_woe_over_time_json(in_data, x_var_list, i, time_var, 'monthly')
        single_chart_data['monitor_frequency']['weekly'] = var_woe_weekly
        single_chart_data['monitor_frequency']['monthly'] = var_woe_monthly
        
        chart_data_list[i] = single_chart_data
    
    return chart_data_list


def bin_distribution_over_time(in_data, x_variable_list, time_variable, frequency = 'DAILY'):
    if type(x_variable_list) == str:
        x_variable_list = [x_variable_list]
    
    in_data = in_data[x_variable_list + [time_variable]].copy()
    
    if frequency.upper() == 'DAILY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 10)
    elif frequency.upper() == 'WEEKLY':
        dates = pd.DataFrame(pd.date_range(start = np.min(in_data[time_variable]), end = np.max(in_data[time_variable]), 
                                           freq = 'D', normalize = True), columns = ['every_day'])
        dates['time_interval'] = dates['every_day'].apply(lambda x: 
            x - datetime.timedelta(days = x.weekday())).astype(str).str.slice(0, 10)
        dates['every_day'] = dates['every_day'].astype(str).str.slice(0, 10)
        in_data['every_day'] = in_data[time_variable].astype(str).str.slice(0, 10)
        in_data = in_data.merge(dates, how = 'left', on = 'every_day').drop('every_day', axis = 1)
    elif frequency.upper() == 'MONTHLY':
        in_data['time_interval'] = in_data[time_variable].astype(str).str.slice(0, 7)
    
    production_dist_json = []
    
    for single_var in x_variable_list:
        in_data['cnt'] = 1
        time_by_var_grouped = in_data.groupby(['time_interval', single_var], as_index = False).agg({'cnt': np.sum})
        time_grouped = in_data.groupby('time_interval', as_index = False).agg({'cnt': np.sum}).rename(columns = {'cnt': 'total_cnt'})
        
        time_by_var_grouped = time_by_var_grouped.merge(time_grouped, how = 'left', on = 'time_interval')
        time_by_var_grouped['var_dist_pct'] = time_by_var_grouped['cnt'] / time_by_var_grouped['total_cnt']
        
        unique_time = pd.Series(time_by_var_grouped['time_interval'].unique()).sort_values()
        unique_bin_value = pd.Series(time_by_var_grouped[single_var].unique()).sort_values()
        
        json_dict = {}
        json_dict['axis_time'] = list(unique_time)
        json_dict['bin_variable'] = single_var
        json_dict['bin_cnt_series'] = {}
        json_dict['bin_dist_series'] = {}
        
        for i in unique_bin_value:
            json_dict['bin_cnt_series'][i] = []
            json_dict['bin_dist_series'][i] = []
        
        for i in unique_time:
            var_df = time_by_var_grouped[time_by_var_grouped['time_interval'] == i]
            
            for j in unique_bin_value:
                if (var_df[var_df[single_var] == j].index.size > 0):
                    if np.isnan(var_df[var_df[single_var] == j].iloc[0]['cnt']):
                        json_dict['bin_cnt_series'][j].append(None)
                    else:
                        json_dict['bin_cnt_series'][j].append(int(var_df[var_df[single_var] == j].iloc[0]['cnt']))
                else:
                    json_dict['bin_cnt_series'][j].append(None)
            
            for j in unique_bin_value:
                if (var_df[var_df[single_var] == j].index.size > 0):
                    if np.isnan(var_df[var_df[single_var] == j].iloc[0]['var_dist_pct']):
                        json_dict['bin_dist_series'][j].append(None)
                    else:
                        json_dict['bin_dist_series'][j].append(
                            float(round(var_df[var_df[single_var] == j].iloc[0]['var_dist_pct'], 4)))
                else:
                    json_dict['bin_dist_series'][j].append(None)
        
        production_dist_json.append(json_dict)
    
    return production_dist_json


def get_refitted_stats_json(in_data, y_variable, x_variables, time_variable):
    in_data = in_data[in_data[y_variable].notnull()][[y_variable, time_variable] + x_variables].copy()
    
    model_data = sm.add_constant(in_data[x_variables], has_constant = 'add')
    
    logit_reg = sm.Logit(in_data[y_variable], model_data)
    result = logit_reg.fit()
    
    vif_dict = {}
    vif_num = 0
    for i in x_variables:
        vif_dict[i] = sm_vif.variance_inflation_factor(np.array(in_data[x_variables]), vif_num)
        vif_num = vif_num + 1
    
    model_stats = {}
    model_stats['dependent_variable'] = y_variable
    model_stats['num_of_obs'] = int(result.nobs)
    model_stats['aic'] = float(round(result.aic, 2))
    model_stats['bic'] = float(round(result.bic, 2))
    
    x_params = list(result.params.index)
    model_stats['x_var_series'] = x_params
    model_stats['coefficient_series'] = []
    model_stats['std_error_series'] = []
    model_stats['z_score_series'] = []
    model_stats['p_value_series'] = []
    model_stats['vif_series'] = []
    
    for i in x_params:
        model_stats['coefficient_series'].append(float(round(dict(result.params)[i], 5)))
        model_stats['std_error_series'].append(float(round(dict(result.bse)[i], 5)))
        model_stats['z_score_series'].append(float(round(dict(result.tvalues)[i], 5)))
        model_stats['p_value_series'].append(float(round(dict(result.pvalues)[i], 5)))
        
        if i in vif_dict.keys():
            model_stats['vif_series'].append(float(round(vif_dict[i], 5)))
        else:
            model_stats['vif_series'].append(None)
    
    PDO = 40
    P0 = 600
    theta0 = 0.08
    B = PDO / math.log(2)
    A = P0 + B * math.log(theta0)
    
    in_data['score'] = A - B * dict(result.params)['const']
    for i in x_variables:
        in_data['score'] = in_data['score'] + -B * dict(result.params)[i] * in_data[i]
    
    model_stats['overview_json'] = get_ks_roc_chart_json(in_data, y_variable, 'score', higher_better = True, 
                                                         score_interval = 10, time_var = time_variable)
    
    return model_stats


def mask_model_json(json_dict):
    index = 1
    x_var_series_new = []
    for i in json_dict['refitted']['x_var_series']:
        if i.replace(' ', '') == 'const':
            x_var_series_new.append(i)
        else:
            x_var_series_new.append('x_variable_' + str(index))
            index = index + 1
    
    json_dict['refitted']['x_var_series'] = x_var_series_new
    
    for i in json_dict['tracker']:
        for j in json_dict['tracker'][i]['monitor_frequency']:
            
            copied_element = json_dict['tracker'][i]['monitor_frequency'][j].copy()
            json_dict['tracker'][i]['monitor_frequency'][j].clear()
            index_bin = 1
            for k in copied_element:
                new_bin_woe_series = {}
                new_bin_dist_series = {}
                index_value = 1
                for m in k['bin_woe_series']:
                    new_bin_woe_series['x_bin_' + str(index_bin) + '_value_' + str(index_value)] = k['bin_woe_series'][m]
                    new_bin_dist_series['x_bin_' + str(index_bin) + '_value_' + str(index_value)] = k['bin_dist_series'][m]
                    index_value = index_value + 1
                
                json_dict['tracker'][i]['monitor_frequency'][j].append({
                    'bin_variable': 'x_bin_' + str(index_bin),
                    'bin_woe_series': new_bin_woe_series,
                    'bin_dist_series': new_bin_dist_series,
                    'iv_series': k['iv_series']})
                index_bin = index_bin + 1
    
    copied_production_dist = json_dict['production_dist'].copy()
    json_dict['production_dist'].clear()
    index_bin = 1
    for i in copied_production_dist:
        new_bin_cnt_series = {}
        new_bin_dist_series = {}
        index_value = 1
        for j in i['bin_cnt_series']:
            if j.lower() == 'error':
                new_bin_cnt_series[j] = i['bin_cnt_series'][j]
                new_bin_dist_series[j] = i['bin_dist_series'][j]
            else:
                new_bin_cnt_series['x_bin_' + str(index_bin) + '_value_' + str(index_value)] = i['bin_cnt_series'][j]
                new_bin_dist_series['x_bin_' + str(index_bin) + '_value_' + str(index_value)] = i['bin_dist_series'][j]
                index_value = index_value + 1
        
        json_dict['production_dist'].append({
            'axis_time': i['axis_time'],
            'bin_variable': 'x_bin_' + str(index_bin),
            'bin_cnt_series': new_bin_cnt_series,
            'bin_dist_series': new_bin_dist_series})
        index_bin = index_bin + 1
    
    return json_dict











