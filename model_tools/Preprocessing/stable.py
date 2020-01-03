"""
A tool to measure variable's distribution between train and test.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class stable_evalutor(object):

    def __init__(self, train, test, target, bin_bun, method, proc_na=True):
        self.train = train
        self.test = test
        self.target = target
        self.bin_bun = bin_bun
        self.method = method
        self.proc_na = proc_na
        self.sel_cols = [x for x in train.columns if x != target]

    @staticmethod
    def get_grouped_data(
            input_data,
            feature,
            target_col,
            bins=8,
            cuts=0,
            method='tree'):
        assert method in ['percentile', 'tree'], "Bin method is unknown."

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
                    next_cut = np.percentile(
                        input_data[feature], i * 100 / bins)
                    if next_cut != prev_cut:
                        cuts.append(next_cut)
                    else:
                        reduced_cuts = reduced_cuts + 1
                    prev_cut = next_cut

                # if reduced_cuts>0:
                #     print('Reduced the number of bins due to less variation in feature')
                cut_series = pd.cut(input_data[feature], cuts)
            elif method == 'tree':
                import scorecardpy as sc
                from scorecardpy.woebin import bins_to_breaks
                bins = sc.woebin(input_data[[feature,
                                             target_col]],
                                 y=target_col,
                                 bin_num_limit=bins,
                                 method="tree",
                                 print_info=False)
                bins_breakslist = bins_to_breaks(bins, input_data)
                cuts = [-np.inf] + \
                    [float(x) for x in bins_breakslist[feature].split(',')] + [np.inf]
                cut_series = pd.cut(input_data[feature], cuts)
        else:
            cut_series = pd.cut(input_data[feature], cuts)

        grouped = input_data.groupby([cut_series], as_index=True).agg(
            {target_col: [np.size, np.mean], feature: [np.mean]})
        grouped.columns = ['_'.join(cols).strip()
                           for cols in grouped.columns.values]
        grouped[grouped.index.name] = grouped.index
        grouped.reset_index(inplace=True, drop=True)
        grouped = grouped[[feature] + list(grouped.columns[0:3])]
        grouped = grouped.rename(
            index=str, columns={
                target_col + '_size': 'Samples_in_bin'})
        grouped = grouped.reset_index(drop=True)
        # corrected_bin_name = '[' + str(min(input_data[feature])) + ', ' + str(grouped.loc[0, feature]).split(',')[1]
        # grouped[feature] = grouped[feature].astype('category')
        # grouped[feature] = grouped[feature].cat.add_categories(corrected_bin_name)
        # grouped.loc[0, feature] = corrected_bin_name

        if has_null == 1:
            grouped_null = grouped.loc[0:0, :].copy()
            grouped_null[feature] = grouped_null[feature].astype('category')
            grouped_null[feature] = grouped_null[feature].cat.add_categories(
                'Nulls')
            grouped_null.loc[0, feature] = 'Nulls'
            grouped_null.loc[0, 'Samples_in_bin'] = len(data_null)
            grouped_null.loc[0, target_col +
                             '_mean'] = data_null[target_col].mean()
            grouped_null.loc[0, feature + '_mean'] = np.nan
            grouped[feature] = grouped[feature].astype('str')
            grouped = pd.concat([grouped_null, grouped], axis=0)
            grouped.reset_index(inplace=True, drop=True)

        grouped[feature] = grouped[feature].astype('str').astype('category')
        if is_train == 1:
            return (cuts, grouped)
        else:
            return (grouped)

    @staticmethod
    def get_trend_changes(grouped_data, feature, target_col, threshold=0.03, proc_na=True):
        if not proc_na:
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

    @staticmethod
    def get_trend_correlation(grouped, grouped_test, feature, target_col, proc_na=True):
        if not proc_na:
            grouped = grouped[grouped[feature] != 'Nulls'].reset_index(drop=True)
            grouped_test = grouped_test[grouped_test[feature] != 'Nulls'].reset_index(drop=True)

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

    @staticmethod
    def get_trend_stats(self, data, target_col, features_list=0, bins=10, data_test=0, method='tree', proc_na=True):
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
                    cuts, grouped = self.get_grouped_data(input_data=data, feature=feature, target_col=target_col, bins=bins,
                                                     method=method)
                    trend_changes = self.get_trend_changes(grouped_data=grouped, feature=feature, target_col=target_col, proc_na=proc_na)
                    if has_test:
                        grouped_test = self.get_grouped_data(input_data=data_test.reset_index(drop=True), feature=feature,
                                                        target_col=target_col, bins=bins, cuts=cuts, method=method)
                        trend_corr = self.get_trend_correlation(grouped, grouped_test, feature, target_col, proc_na=proc_na)
                        trend_changes_test = self.get_trend_changes(grouped_data=grouped_test, feature=feature,
                                                               target_col=target_col, proc_na=proc_na)
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

        @property
        def stats(self):
            stats = self.get_trend_stats(data=self.train,
            target_col=self.target,
            bins=8,
            data_test=self.test,
            method='tree',
            proc_na=True)
        return stats_all_df
