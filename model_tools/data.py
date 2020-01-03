# -*- coding: utf-8 -*-

"""
DataHelper

"""
import pandas as pd
from .utils import is_numpy


class DataHelper(object):
    def __init__(self, target, train_path, test_path, trainfile=None, testfile=None, date_cols=None):

        assert not ((trainfile is None) and (train_path is None)), "trainfile or train_path at least one is set"
        assert not ((trainfile is not None) and (train_path is not None)), "only one can be set"
        # assert not ((testfile is None) and (test_path is None)), "testfile or test_path at least one is set"
        # assert not ((testfile is not None) and (test_path is not None)), "only one can be set"

        self.target = target
        self.train_path = train_path
        self.test_path = test_path
        self.train = trainfile
        self.test = testfile
        self.date_cols = date_cols

        self.ntrain = None
        self.continues_features = None
        self.object_features = None
        self.sel_cols = None

    def load_data(self):
        file_type = self.train_path.split('/')[-1].split('.')[-1]

        assert file_type in ['csv', 'xlsx', 'pkl', 'json'], "DataHelper don't support the input file type! "

        if file_type == 'csv':
            try:
                self.train = pd.read_csv(self.train_path)
                if self.test_path is not None:
                    self.test = pd.read_csv(self.test_path)
            except UnicodeDecodeError:
                self.train = pd.read_csv(self.train_path, encoding='GBK')
                if self.test_path is not None:
                    self.test = pd.read_csv(self.test_path, encoding='GBK')
        elif file_type == 'xlsx':
            self.train = pd.read_excel(self.train_path)
            if self.test_path is not None:
                self.test = pd.read_excel(self.test_path)
        elif file_type == 'pkl':
            self.train = pd.read_pickle(self.train_path)
            if self.test_path is not None:
                self.test = pd.read_pickle(self.test_path)
        elif file_type == 'json':
            self.train = pd.read_json(self.train_path)
            if self.test_path is not None:
                self.test = pd.read_json(self.test_path)

        self.ntrain = self.train.shape[0]

        if self.test is not None and self.target not in self.test.columns:
            self.test[self.target] = -1

        sel_cols = [f_ for f_ in self.train.columns if f_ in self.test.columns] \
            if self.test is not None else [f_ for f_ in self.train.columns]
        self.sel_cols = sel_cols
        dtypes = self.train[sel_cols].dtypes

        self.object_features = [f_ for f_ in dtypes[dtypes.isin(['category', 'object'])].index.tolist()] if \
            self.date_cols is None else [f_ for f_ in dtypes[dtypes.isin(['category', 'object'])].index.tolist()
                                         if f_ not in self.date_cols]

        self.continues_features = dtypes[dtypes.apply(lambda x: True if str(x).startswith('int') or str(x).
                                                      startswith('float') else False)].index.tolist()
        return self.train, self.test

    def combine(self):
        if self.train_path is not None:
            self.load_data()

        self.ntrain = self.train.shape[0]

        if self.test is not None and self.target not in self.test.columns:
            self.test[self.target] = -1

        sel_cols = [f_ for f_ in self.train.columns if f_ in self.test.columns] \
            if self.test is not None else [f_ for f_ in self.train.columns]
        self.sel_cols = sel_cols
        dtypes = self.train[sel_cols].dtypes

        self.object_features = [f_ for f_ in dtypes[dtypes.isin(['category', 'object'])].index.tolist()] if \
            self.date_cols is None \
            else [f_ for f_ in dtypes[dtypes.isin(['category', 'object'])].index.tolist() if f_ not in self.date_cols]

        self.continues_features = dtypes[dtypes.apply(lambda x: True if str(x).startswith('int') or str(x).
                                                      startswith('float') else False)].index.tolist()
        if self.test is not None:
            return pd.concat((self.train[self.sel_cols], self.test[self.sel_cols]), axis=0).reset_index(drop=True)

    def split(self, train_test):
        if self.ntrain is None:
            return None
        if is_numpy(train_test):
            train = train_test[:self.ntrain, :]
            test = train_test[self.ntrain:, :]
        else:
            train = train_test.iloc[:self.ntrain, :].copy().reset_index(drop=True)
            test = train_test.iloc[self.ntrain:, :].copy().reset_index(drop=True)
        return train, test
