"""
A tool to measure variable's distribution between train and test.
Add binning method
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scorecardpy as sc
from scorecardpy.woebin import bins_to_breaks


def get_grouped_data(input_data, feature, target_col, bins, cuts=0, method='tree'):
    """
    Bins continuous features into equal sample size buckets and returns the target mean in each bucket. Separates out
    nulls into another bucket.
    :param input_data: dataframe containg features and target column
    :param feature: feature column name
    :param target_col: target column
    :param bins: Number bins required
    :param cuts: if buckets of certain specific cuts are required. Used on test data to use cuts from train.
    :return: If cuts are passed only grouped data is returned, else cuts and grouped data is returned
    """
    has_null = pd.isnull(input_data[feature]).sum() > 0
    if has_null == 1:
        data_null = input_data[pd.isnull(input_data[feature])]
        input_data = input_data[~pd.isnull(input_data[feature])]
        input_data.reset_index(inplace=True, drop=True)

    is_train = 0
    if cuts == 0:
        is_train = 1
        if method == 'percentile':
            prev_cut = min(input_data[feature]) - 1
            cuts = [prev_cut]
            reduced_cuts = 0
            for i in range(1, bins + 1):
                next_cut = np.percentile(input_data[feature], i * 100 / bins)
                if next_cut != prev_cut:
                    cuts.append(next_cut)
                else:
                    reduced_cuts = reduced_cuts + 1
                prev_cut = next_cut

            # if reduced_cuts>0:
            #     print('Reduced the number of bins due to less variation in feature')
            cut_series = pd.cut(input_data[feature], cuts)
        elif method == 'tree':
            bins = sc.woebin(input_data[[feature, target_col]], y=target_col, bin_num_limit=bins, method="tree", print_info=False)
            bins_breakslist = bins_to_breaks(bins, input_data)
            cuts = [-np.inf] + [float(x) for x in bins_breakslist[feature].split(',')] + [np.inf]
            cut_series = pd.cut(input_data[feature], cuts)
    else:
        cut_series = pd.cut(input_data[feature], cuts)

    grouped = input_data.groupby([cut_series], as_index=True).agg(
        {target_col: [np.size, np.mean], feature: [np.mean]})
    grouped.columns = ['_'.join(cols).strip() for cols in grouped.columns.values]
    grouped[grouped.index.name] = grouped.index
    grouped.reset_index(inplace=True, drop=True)
    grouped = grouped[[feature] + list(grouped.columns[0:3])]
    grouped = grouped.rename(index=str, columns={target_col + '_size': 'Samples_in_bin'})
    grouped = grouped.reset_index(drop=True)
    #corrected_bin_name = '[' + str(min(input_data[feature])) + ', ' + str(grouped.loc[0, feature]).split(',')[1]
    #grouped[feature] = grouped[feature].astype('category')
    #grouped[feature] = grouped[feature].cat.add_categories(corrected_bin_name)
    #grouped.loc[0, feature] = corrected_bin_name

    if has_null == 1:
        grouped_null = grouped.loc[0:0, :].copy()
        grouped_null[feature] = grouped_null[feature].astype('category')
        grouped_null[feature] = grouped_null[feature].cat.add_categories('Nulls')
        grouped_null.loc[0, feature] = 'Nulls'
        grouped_null.loc[0, 'Samples_in_bin'] = len(data_null)
        grouped_null.loc[0, target_col + '_mean'] = data_null[target_col].mean()
        grouped_null.loc[0, feature + '_mean'] = np.nan
        grouped[feature] = grouped[feature].astype('str')
        grouped = pd.concat([grouped_null, grouped], axis=0)
        grouped.reset_index(inplace=True, drop=True)

    grouped[feature] = grouped[feature].astype('str').astype('category')
    if is_train == 1:
        return (cuts, grouped)
    else:
        return (grouped)


def draw_plots(input_data, feature, target_col, trend_correlation=None):
    """
    Draws univariate dependence plots for a feature
    :param input_data: grouped data contained bins of feature and target mean.
    :param feature: feature column name
    :param target_col: target column
    :param trend_correlation: correlation between train and test trends of feature wrt target
    :return: Draws trend plots for feature
    """
    trend_changes = get_trend_changes(grouped_data=input_data, feature=feature, target_col=target_col)
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(input_data[target_col + '_mean'], marker='o')
    ax1.set_xticks(np.arange(len(input_data)))
    ax1.set_xticklabels((input_data[feature]).astype('str'))
    plt.xticks(rotation=45)
    ax1.set_xlabel('Bins of ' + feature)
    ax1.set_ylabel('Average of ' + target_col)
    comment = "Trend changed " + str(trend_changes) + " times"
    if trend_correlation == 0:
        comment = comment + '\n' + 'Correlation with train trend: NA'
    elif trend_correlation != None:
        comment = comment + '\n' + 'Correlation with train trend: ' + str(int(trend_correlation * 100)) + '%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.05, 0.95, comment, fontsize=12, verticalalignment='top', bbox=props, transform=ax1.transAxes)
    plt.title('Average of ' + target_col + ' wrt ' + feature)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(np.arange(len(input_data)), input_data['Samples_in_bin'], alpha=0.5)
    ax2.set_xticks(np.arange(len(input_data)))
    ax2.set_xticklabels((input_data[feature]).astype('str'))
    plt.xticks(rotation=45)
    ax2.set_xlabel('Bins of ' + feature)
    ax2.set_ylabel('Bin-wise sample size')
    plt.title('Samples in bins of ' + feature)
    plt.tight_layout()
    plt.show()


def get_trend_changes(grouped_data, feature, target_col, threshold=0.03):
    """
    Calculates number of times the trend of feature wrt target changed direction.
    :param grouped_data: grouped dataset
    :param feature: feature column name
    :param target_col: target column
    :param threshold: minimum % difference required to count as trend change
    :return: number of trend chagnes for the feature
    """
    grouped_data = grouped_data.loc[grouped_data[feature] != 'Nulls', :].reset_index(drop=True)
    target_diffs = grouped_data[target_col + '_mean'].diff()
    target_diffs = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff = grouped_data[target_col + '_mean'].max() - grouped_data[target_col + '_mean'].min()
    target_diffs_mod = target_diffs.fillna(0).abs()
    low_change = target_diffs_mod < threshold * max_diff
    target_diffs_norm = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2 = target_diffs_norm.diff()
    changes = target_diffs_lvl2.fillna(0).abs() / 2
    tot_trend_changes = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return (tot_trend_changes)


def get_trend_correlation(grouped, grouped_test, feature, target_col):
    """
    Calculates correlation between train and test trend of feature wrt target.
    :param grouped: train grouped data
    :param grouped_test: test grouped data
    :param feature: feature column name
    :param target_col: target column name
    :return: trend correlation between train and test
    """
    # grouped = grouped[grouped[feature] != 'Nulls'].reset_index(drop=True)
    # grouped_test = grouped_test[grouped_test[feature] != 'Nulls'].reset_index(drop=True)

    if grouped_test.loc[0, feature] != grouped.loc[0, feature]:
        grouped_test[feature] = grouped_test[feature].cat.add_categories(grouped.loc[0, feature])
        grouped_test.loc[0, feature] = grouped.loc[0, feature]
    grouped_test_train = grouped.merge(grouped_test[[feature, target_col + '_mean']], on=feature, how='left',
                                       suffixes=('', '_test'))
    nan_rows = pd.isnull(grouped_test_train[target_col + '_mean']) | pd.isnull(
        grouped_test_train[target_col + '_mean_test'])
    grouped_test_train = grouped_test_train.loc[~nan_rows, :]
    if len(grouped_test_train) > 1:
        trend_correlation = np.corrcoef(grouped_test_train[target_col + '_mean'],
                                        grouped_test_train[target_col + '_mean_test'])[0, 1]
    else:
        trend_correlation = 0
        print("Only one bin created for " + feature + ". Correlation can't be calculated")

    return (trend_correlation)


def univariate_plotter(feature, data, target_col, bins=10, data_test=0):
    """
    Calls the draw plot function and editing around the plots
    :param feature: feature column name
    :param data: dataframe containing features and target columns
    :param target_col: target column name
    :param bins: number of bins to be created from continuous feature
    :param data_test: test data which has to be compared with input data for correlation
    :return: grouped data if only train passed, else (grouped train data, grouped test data)
    """
    print(' {:^100} '.format('Plots for ' + feature))
    if data[feature].dtype == 'O':
        print('Categorical feature not supported')
    else:
        cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col, bins=bins)
        has_test = type(data_test) == pd.core.frame.DataFrame
        if has_test:
            grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True), feature=feature,
                                            target_col=target_col, bins=bins, cuts=cuts)
            trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
            print(' {:^100} '.format('Train data plots'))

            draw_plots(input_data=grouped, feature=feature, target_col=target_col)
            print(' {:^100} '.format('Test data plots'))

            draw_plots(input_data=grouped_test, feature=feature, target_col=target_col, trend_correlation=trend_corr)
        else:
            draw_plots(input_data=grouped, feature=feature, target_col=target_col)
        print(
            '---------------------------------------------------------------------------------------------------------')
        print('\n')
        if has_test:
            return (grouped, grouped_test)
        else:
            return (grouped)


def get_univariate_plots(data, target_col, features_list=0, bins=10, data_test=0):
    """
    Creates univariate dependence plots for features in the dataset
    :param data: dataframe containing features and target columns
    :param target_col: target column name
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
    :param bins: number of bins to be created from continuous feature
    :param data_test: test data which has to be compared with input data for correlation
    :return: Draws univariate plots for all columns in data
    """
    if type(features_list) == int:
        features_list = list(data.columns)
        features_list.remove(target_col)

    for cols in features_list:
        if cols != target_col and data[cols].dtype == 'O':
            print(cols + ' is categorical. Categorical features not supported yet.')
        elif cols != target_col and data[cols].dtype != 'O':
            univariate_plotter(feature=cols, data=data, target_col=target_col, bins=bins, data_test=data_test)


def get_trend_stats(data, target_col, features_list=0, bins=10, data_test=0, method='tree'):
    """
    Calculates trend changes and correlation between train/test for list of features
    :param data: dataframe containing features and target columns
    :param target_col: target column name
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
    :param bins: number of bins to be created from continuous feature
    :param data_test: test data which has to be compared with input data for correlation
    :return: dataframe with trend changes and trend correlation (if test data passed)
    """

    if type(features_list) == int:
        features_list = list(data.columns)
        features_list.remove(target_col)

    stats_all = []
    has_test = type(data_test) == pd.core.frame.DataFrame
    ignored = []
    for feature in features_list:
        try:
            if data[feature].dtype == 'O' or feature == target_col:
                ignored.append(feature)
            else:
                cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col, bins=bins, method=method)
                trend_changes = get_trend_changes(grouped_data=grouped, feature=feature, target_col=target_col)
                if has_test:
                    grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True), feature=feature,
                                                    target_col=target_col, bins=bins, cuts=cuts, method=method)
                    trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
                    trend_changes_test = get_trend_changes(grouped_data=grouped_test, feature=feature,
                                                           target_col=target_col)
                    stats = [feature, trend_changes, trend_changes_test, trend_corr]
                else:
                    stats = [feature, trend_changes]
                stats_all.append(stats)
        except:
            continue
    stats_all_df = pd.DataFrame(stats_all)
    try:
        stats_all_df.columns = ['Feature', 'Trend_changes'] if has_test == False else ['Feature', 'Trend_changes',
                                                                                   'Trend_changes_test',
                                                                                   'Trend_correlation']
    except:
        pass

    if len(ignored) > 0:
        print('Categorical features ' + str(ignored) + ' ignored. Categorical features not supported yet.')

    print('Returning stats for all numeric features')
    return stats_all_df


# calculate CSI & PSI
def csi(ex_var, ac_var):
    """calculate csi of variable
    Input:
        1. ex_var: expected variable woe
        2. ac_var: actual variable woe
    Output:
        CSI value
    """
    res = ex_var.value_counts().to_frame()
    res.columns = ['expected']
    res = res.join(ac_var.value_counts().to_frame(), how='outer')
    res.columns = ['expected', 'actual']
    res.fillna(1, inplace=True)
    res = res / res.sum()
    res['psi'] = (res['expected'] / res['actual']).apply(lambda x: np.log(x))
    res['psi'] = (res['expected'] - res['actual']) * res['psi']
    res['expected'] = res['expected'].apply(lambda x: round(x, 6))
    res['actual'] = res['actual'].apply(lambda x: round(x, 6))
    res['psi'] = res['psi'].apply(lambda x: round(x, 6))

    return res


# 等频
def psi(ex_score, ac_score, segments=10):
    """calculate psi of model scoring
    Input:
        1. ex_score: expected score
        2. ac_score: actual score
    Output:
        PSI value
    """
    interval = 1.0 / segments
    seg = ex_score.quantile(np.arange(0, 1, interval)).tolist()
    percent_list = []
    score_range = []

    for i in range(segments):
        if i == 0:
            ex_percent_series = ex_score.apply(lambda x: x <= seg[i + 1])
            ac_percent_series = ac_score.apply(lambda x: x <= seg[i + 1])
            score_range.append('<={0}'.format(round(seg[i + 1])))
        elif i == segments - 1:
            ex_percent_series = ex_score.apply(lambda x: x > seg[i])
            ac_percent_series = ac_score.apply(lambda x: x > seg[i])
            score_range.append('>{0}'.format(round(seg[i])))
        else:
            ex_percent_series = ex_score.apply(lambda x: seg[i] < x <= seg[i + 1])
            ac_percent_series = ac_score.apply(lambda x: seg[i] < x <= seg[i + 1])
            score_range.append('{0}-{1}'.format(round(seg[i]), round(seg[i + 1])))

        ex_percent = max(float(ex_percent_series.sum()), 1.0) / ex_percent_series.count()
        ac_percent = max(float(ac_percent_series.sum()), 1.0) / ac_percent_series.count()
        percent_list.append((ex_percent, ac_percent))

    total_psi = 0
    psi_list = []
    for p in percent_list:
        psi = (p[1] - p[0]) * np.log(p[1] / p[0])
        psi_list.append(psi)
        total_psi += psi

    report_df = pd.DataFrame(columns=['score_range', 'expected_percent',
                                      'actual_percent', 'psi'])
    report_df['score_range'] = score_range
    report_df[['expected_percent', 'actual_percent']] = percent_list
    report_df['psi'] = psi_list

    return report_df


# 等距
def psi_v2(ex_score, ac_score, quant=10):
    minv = min(min(ac_score), min(ex_score))
    maxv = max(max(ac_score), max(ex_score))
    step = 1.0 * (maxv - minv) / quant
    segp = []
    for i in range(quant):
        if i == 0:
            sp = minv
            segp.append(sp)
        else:
            sp = sp + step
            segp.append((sp))

    percent_list = []
    score_range = []
    for i in range(quant):
        if i == 0:
            ex_percent_series = ex_score.apply(lambda x: x <= segp[i + 1])
            ac_percent_series = ac_score.apply(lambda x: x <= segp[i + 1])
            score_range.append('<={0}'.format(round(segp[i + 1])))
        elif i == quant - 1:
            ex_percent_series = ex_score.apply(lambda x: x > segp[i])
            ac_percent_series = ac_score.apply(lambda x: x > segp[i])
            score_range.append('>{0}'.format(round(segp[i])))
        else:
            ex_percent_series = ex_score.apply(lambda x: segp[i] < x <= segp[i + 1])
            ac_percent_series = ac_score.apply(lambda x: segp[i] < x <= segp[i + 1])
            score_range.append('{0}-{1}'.format(round(segp[i]), round(segp[i + 1])))

        ex_percent = max(float(ex_percent_series.sum()), 1.0) / ex_percent_series.count()
        ac_percent = max(float(ac_percent_series.sum()), 1.0) / ac_percent_series.count()
        percent_list.append((ex_percent, ac_percent))

    total_psi = 0
    psi_list = []
    for p in percent_list:
        psi = (p[1] - p[0]) * np.log(p[1] / p[0])
        psi_list.append(psi)
        total_psi += psi

    report_df = pd.DataFrame(columns=['score_range', 'expected_percent',
                                      'actual_percent', 'psi'])
    report_df['score_range'] = score_range
    report_df[['expected_percent', 'actual_percent']] = percent_list
    report_df['psi'] = psi_list
    return report_df
