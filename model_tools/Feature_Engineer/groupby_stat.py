# -*- coding: utf-8 -*-

"""
generate static features though groupby categorical features and aggregate continues vars.

@author: Moon
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupbyStaticMethod(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        """
        Parameter
        ---------
        params: params is a dict contain groupby vars、 aggregated var and static method
                params key is categorical features(one or more, default sep: '__')
                params value is list that contains one or more dict(key: aggregated vars| values: static method)
        Example
        --------
        params = {'groupby1':{'agg_var1':'mean'},
                  'groupby2':{'agg_var2':'std'},
                  'groupby1__groupby2':{'agg_var3': ['max', 'count']}}
        """
        self.params = params
        self.static_dict = None
        self.new_name_tuple = []

    def fit(self, X, y=0):
        self.static_dict = {}
        import operator
        self.params=sorted(self.params.items(),key=operator.itemgetter(0))
        init = pd.DataFrame()
        for k, v in self.params:
            if str(k).find('#') != -1:
                k = str(k).split('#')[0]

            if str(k).find('&') != -1:
                by = [x for x in str(k).split('&')]
            else:
                by = str(k)
            if str(by) not in init.columns.tolist():
                init = pd.DataFrame(X[by].drop_duplicates().reset_index(drop=True))
            for target, method in v.items():
                if type(method) == list:
                    for mtd in method:
                        rename = mtd + '_' + target + '_by_' + '&'.join(by) if type(by) == list else \
                            mtd + '_' + target + '_by_' + by
                        if mtd in ['mean', 'var', 'std']:
                            self.new_name_tuple.append(tuple((target, rename)))
                        tmp = X.groupby(by)[target].agg({rename: lambda x: x.nunique()} if mtd == 'nunique' else
                                                        {rename: mtd}).reset_index()
                        init = pd.merge(init, tmp, on=by, how='left')
                elif type(method) == str:
                    rename = method + '_' + target + '_by_' + '&'.join(by) if type(by) == list else \
                        method + '_' + target + '_by_' + by
                    tmp = X.groupby(by)[target].agg({rename: lambda x: x.nunique()} if method == 'nunique' else
                                                    {rename: method}).reset_index()
                    init = pd.merge(init, tmp, on=by, how='left')
            self.static_dict[k] = init
        return self

    def fit_transform(self, X, y=0):
        self.fit(X, y=0)
        return self.transform(X, y=0)

    def transform(self, X, y=0):

        for k in self.static_dict.keys():
            if str(k).find('&') != -1:
                by = [x for x in str(k).split('&')]
            else:
                by = str(k)

            init = self.static_dict[k]
            X = pd.merge(X, init, on=by, how='left')
          
        for col, new_col in self.new_name_tuple:
            if 'mean' in new_col:
                X[col + '_dif_' + new_col] = X[col] - X[new_col]
                X[col + '_div_' + new_col] = X[col + '_dif_' + new_col]/(X[col] + 1e-7)
            elif 'count' in new_col:
                X[col + '_div_' + new_col] = X[new_col]/(X['cnt_enc_'+col] + 1e-7)
        
        return X
