#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score card model utils
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from functools import wraps
import sys
indent = "    "


#def eprint(*args, **kwargs):
#    print(*args, file=sys.stderr, **kwargs)
    

def logit_fit(df, y_variable, x_variables):
    model_data = sm.add_constant(df.loc[:, x_variables])
    logit_reg = sm.Logit(df[y_variable], model_data)
    try:
        result = logit_reg.fit(disp=0)
    except:
        result = logit_reg.fit(disp=0, method='bfgs')
    #result = logit_reg.fit()
    print(result.summary2())
    model_data = model_data.merge(df[[y_variable]], how='left',
                                  left_index=True, right_index=True)
    model_data['prob'] = logit_reg.predict(result.params)
    predit_data = model_data.loc[:, [y_variable, 'prob']]
    return result.params, predit_data


def logit_predict(df, y_variable, x_variables, model_params):
    model_data = sm.add_constant(df[x_variables])
    logit_reg = sm.Logit(df[y_variable], model_data)
    model_data = model_data.merge(df[[y_variable]], how='left', left_index=True,
                                  right_index=True)
    model_data['prob'] = logit_reg.predict(model_params)
    predit_data = model_data.loc[:, [y_variable, 'prob']]
    return predit_data


def greater_than(a, b):
    return 1 if a > b else 0


def split_df(df, column=None, pct=0.3):
    """
    切割DataFrame
    """
    df_dict = {}
    if column is None:
        train, test = train_test_split(df, test_size=pct)
        df_dict['train'] = train
        df_dict['test'] = test
    else:
        df[column] = df[column].fillna('None_Type')
        for i in df[column]:
            df_dict[i] = df.loc[df[column] == i]
    return df_dict


class ModelPlot(object):
    def __init__(self, df, score='prob', y_variable='npd30', pos_label=1, subplot_num=2, title=''):
        self.data = df.loc[pd.notnull(df[y_variable])]
        self.data[score] = self.data[score].astype(float)
        self.data[y_variable] = self.data[y_variable].astype(int)
        self.score = score
        self.y_variable = y_variable
        self.pos_label = pos_label  # Y变量的正例值
        self.subplot_num = subplot_num
        self.false_positive_rate, self.recall, self.thresholds = roc_curve(self.data[self.y_variable],
                                                                           self.data[self.score], self.pos_label)
        self.roc_auc = auc(self.false_positive_rate, self.recall)
        self.ks = abs(self.false_positive_rate - self.recall).max()
        self.title = title
        self.vfunc = np.vectorize(greater_than)

    def get_cum_group(self):
        data = self.data
        data['good'] = 1 - data[self.y_variable]
        data_grouped = data[[self.score, self.y_variable, 'good']].groupby(self.score).agg(
            {self.y_variable: np.sum, 'good': np.sum}).reset_index()
        data_grouped['cum_pct_0'] = data_grouped['good'].cumsum() / (data.shape[0] - np.sum(data[self.y_variable]))
        data_grouped['cum_pct_1'] = data_grouped[self.y_variable].cumsum() / np.sum(data[self.y_variable])
        data_grouped['ks'] = abs((data_grouped['cum_pct_0'] - data_grouped['cum_pct_1']) * 100)
        return data_grouped

    def get_bad_group(self, bins_count=20):
        score_df = self.data.loc[:, [self.score, self.y_variable]]
        score_df['bins'] = pd.qcut(score_df[self.score], bins_count)
        group_df = score_df.groupby('bins')[self.y_variable].value_counts().unstack(level=-1)
        group_df.columns = ['good', 'bad']
        group_df_plot = group_df.loc[:, ['good', 'bad']]
        group_df_stack = group_df_plot.div(group_df_plot.sum(1).astype(float), axis=0)
        return group_df_stack['bad']

    def roc_plot(self):
        """
        matplotlib画roc
        """
        fig1 = plt.figure(1, figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.false_positive_rate, self.recall)
        ax1.text(0.5, 0.34, "AUC Score = %.3f" % self.roc_auc)
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.axis([0.0, 1.0, 0.0, 1.0])
        ax1.set_title(self.title + " ROC Curve")
        ax1.fill_between(self.false_positive_rate, self.recall, alpha=0.3, linewidth=0, )
        plt.show()

    def ks_plot(self):
        """
        matplotlib画ks
        """
        data_grouped = self.get_cum_group()
        max_ks = data_grouped.loc[data_grouped['ks'].idxmax(): data_grouped['ks'].idxmax()]
        fig2 = plt.figure(2, figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        ax2.axis([0.0, data_grouped[self.score].max(), 0.0, 1.0])
        ax2.plot(data_grouped[self.score], data_grouped['cum_pct_0'], linewidth=2)
        ax2.plot(data_grouped[self.score], data_grouped['cum_pct_1'], linewidth=2)
        ax2.plot([max_ks[self.score].iloc[-1], max_ks[self.score].iloc[-1]],
                 [max_ks['cum_pct_0'].iloc[-1], max_ks['cum_pct_1'].iloc[-1]], 'k--', lw=2)
        ax2.set_title(self.title + " KS Curve")
        ax2.set_ylabel("Cumulative")
        ax2.set_ｘlabel("Score")
        ax2.annotate(str(round(max_ks['cum_pct_1'].iloc[-1], 2)),
                     xy=(max_ks[self.score].iloc[-1] + 0.01 * data_grouped[self.score].max(),
                         max_ks['cum_pct_1'].iloc[-1]), fontsize=14)
        ax2.annotate(str(round(max_ks['cum_pct_0'].iloc[-1], 2)),
                     xy=(max_ks[self.score].iloc[-1] - 0.08 * data_grouped[self.score].max(),
                         max_ks['cum_pct_0'].iloc[-1]), fontsize=14)
        ax2.text(0.6 * np.max(data_grouped[self.score]), np.min(data_grouped['cum_pct_0']) + 0.34,
                 'KS = %.3f' % max_ks['ks'].iloc[-1])
        plt.show()

    def badrate_plot(self, bins_count=20):
        bad_group = self.get_bad_group(bins_count=bins_count)
        plt.figure(3, figsize=(8, 6))
        ax = bad_group.plot(kind='barh', stacked=True,
                            xlim=[0, 1.2 * np.max(bad_group)],
                            title=self.title + ' BadRate of Score Distribution')
        x_offset = 0.001
        ax.set_xlabel("BadRate")
        for p in ax.patches:
            b = p.get_bbox()
            val = "{:.2%}".format(b.x1)
            ax.annotate(val, (b.x1 + x_offset, b.y0), fontsize=12)
        plt.show()

    def model_plot(self, bins_count=20):
        """
        matplotlib画ks,roc
        """
        data_grouped = self.get_cum_group()
        bad_group = self.get_bad_group(bins_count=bins_count)
        max_ks = data_grouped.loc[data_grouped['ks'].idxmax(): data_grouped['ks'].idxmax()]
        fig = plt.figure(figsize=(23, 6))
        ax1 = fig.add_subplot(1, 3, 3)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 1)
        ax1.plot(self.false_positive_rate, self.recall)
        ax1.text(0.5, 0.34, "AUC Score = %.3f" % self.roc_auc)
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.axis([0.0, 1.0, 0.0, 1.0])
        ax1.set_title(self.title + " ROC Curve")
        ax1.fill_between(self.false_positive_rate, self.recall, alpha=0.3, linewidth=0, )
        ax2.axis([0.0, data_grouped[self.score].max(), 0.0, 1.0])
        ax2.plot(data_grouped[self.score], data_grouped['cum_pct_0'], linewidth=2)
        ax2.plot(data_grouped[self.score], data_grouped['cum_pct_1'], linewidth=2)
        ax2.plot([max_ks[self.score].iloc[-1], max_ks[self.score].iloc[-1]],
                 [max_ks['cum_pct_0'].iloc[-1], max_ks['cum_pct_1'].iloc[-1]], 'k--', lw=2)
        ax2.annotate(str(round(max_ks['cum_pct_1'].iloc[-1], 2)),
                     xy=(max_ks[self.score].iloc[-1] + 0.01 * data_grouped[self.score].max(),
                         max_ks['cum_pct_1'].iloc[-1]), fontsize=14)
        ax2.annotate(str(round(max_ks['cum_pct_0'].iloc[-1], 2)),
                     xy=(max_ks[self.score].iloc[-1] - 0.08 * data_grouped[self.score].max(),
                         max_ks['cum_pct_0'].iloc[-1]), fontsize=14)
        ax2.set_title(self.title + " KS Curve")
        ax2.set_ylabel("Cumulative")
        ax2.set_ｘlabel("Score")
        ax2.text(0.6 * np.max(data_grouped[self.score]), np.min(data_grouped['cum_pct_0']) + 0.34,
                 'KS = %.3f' % max_ks['ks'].iloc[-1])
        bad_group.plot(ax=ax3, kind='barh', stacked=True,
                       xlim=[0, 1.2 * np.max(bad_group)],
                       title=self.title + ' BadRate of Score Distribution')
        ax3.set_xlabel("BadRate")
        x_offset = 0.001
        for p in ax3.patches:
            b = p.get_bbox()
            val = "{:.2%}".format(b.x1)
            ax3.annotate(val, (b.x1 + x_offset, b.y0), fontsize=12)
        plt.subplots_adjust(wspace=0.2)
        plt.show()

    def pdf_plot(self):
        score_df = self.data.loc[:, [self.score, self.y_variable]]
        good = score_df.loc[score_df[self.y_variable] == 0][self.score]
        good.name = 'Goods'
        bad = score_df.loc[score_df[self.y_variable] == 1][self.score]
        bad.name = 'Bads'
        fig4 = plt.figure(4, figsize=(8, 6))
        ax4 = fig4.add_subplot(111)
        ax4.set_xlabel("Score")
        sns.despine(left=True)
        ax4.set_title(self.title + " PDF Between Goods and Bads")
        sns.kdeplot(good, shade=True, color="#B14C4D", ax=ax4)
        sns.kdeplot(bad, shade=True, color="#577A97", ax=ax4)
        plt.show()

    def cm_plot(self, cut_off=0.5):
        """
        :param cut_off: 默认阈值0.5
        """
        y_pred = self.vfunc(self.data[self.score], cut_off)
        cm = confusion_matrix(self.data[self.y_variable], y_pred)
        fig5 = plt.figure(5, figsize=(8, 6))
        ax5 = fig5.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax5)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def pr_plot(self):
        precision, recall, threshold = precision_recall_curve(self.data[self.y_variable],
                                                              self.data[self.score])
        fig = plt.figure(6, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.axis([0.0, 1.0, 0.0, 1.0])
        ax.set_title(self.title + " Precision Vs Recall Curve")
        ax.fill_between(precision, recall, alpha=0.3, linewidth=0, )
        plt.show()

    def pc_plot(self):
        """
        查看prob的适合的Cutoff
        """
        max_t = max(self.data[self.score])
        cut = []
        prec = []
        for i in range(0, 101, 1):
            if i / 100. <= max_t:
                cut.append(i / 100.)
                prec.append(precision_score(self.data[self.y_variable],
                                            self.vfunc(self.data[self.score], cut[i])))

        fig = plt.figure(7, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(cut, prec)
        ax.set_xlabel("Cut-Off")
        ax.set_ylabel("Precision")
        ax.set_title(self.title + " Precision Vs Cut-off Curve")
        plt.show()

    def rc_plot(self):
        max_t = max(self.data[self.score])
        cut = []
        rec = []
        for i in range(0, 101, 1):
            if i / 100. <= max_t:
                cut.append(i / 100.)
                rec.append(recall_score(self.data[self.y_variable],
                                        self.vfunc(self.data[self.score], cut[i])))
        fig = plt.figure(8, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(cut, rec)
        ax.set_xlabel("Cut-Off")
        ax.set_ylabel("Recall")
        ax.set_title(self.title + " Recall Vs Cut-off Curve")
        plt.show()


def quick_fit_plot(data_dict,x_variables,y_variable='npd30'):
    """
    data_dict的key必须是train,valid,test;或者train,valid
    """
    if len(data_dict)==3:
        print ("    ##################### Train Stats #########################")
        train_df,valid_df,test_df =  data_dict['train'],data_dict['valid'],data_dict['test']
        train_param,train_predict_data = logit_fit(train_df,y_variable,x_variables)
        train_plot = ModelPlot(train_predict_data,y_variable=y_variable,title='Train')
        train_plot.roc_plot()
        train_plot.ks_plot()
        print('\n')
        print ("    ##################### Valid Stats #########################")
        valid_predict_data = logit_predict(valid_df,y_variable,x_variables,train_param)
        valid_plot = ModelPlot(valid_predict_data,y_variable=y_variable,title='Valid')
        valid_plot.roc_plot()
        valid_plot.ks_plot()
        print('\n')
        print ("    ##################### Test Stats #########################")
        test_predict_data = logit_predict(test_df,y_variable,x_variables,train_param)
        test_plot = ModelPlot(test_predict_data,y_variable=y_variable,title='Test')
        test_plot.roc_plot()
        test_plot.ks_plot()
    elif len(data_dict)==2:
        print ("    ##################### Train Stats #########################")
        train_df,valid_df =  data_dict['train'],data_dict['valid']
        train_param,train_predict_data = logit_fit(train_df,y_variable,x_variables)
        train_plot = ModelPlot(train_predict_data,y_variable=y_variable,title='Train')
        train_plot.roc_plot()
        train_plot.ks_plot()
        print('\n')
        print ("    ##################### Valid Stats #########################")
        valid_predict_data = logit_predict(valid_df,y_variable,x_variables,train_param)
        valid_plot = ModelPlot(valid_predict_data,y_variable=y_variable,title='Valid')
        valid_plot.roc_plot()
        valid_plot.ks_plot()
    else:
        print('请看我的注释')



"""
下面是StepWise
"""
class LRStats(object):
    """此类打印中间结果的相关信息，并保存中间结果"""

    def __init__(self, step, n, p, res):
        self.res = res
        self.aic = res.aic
        self.bic = res.bic
        self.logl = -2 * res.llf
        self.sc = 2 * (-res.llf + p * (np.log(n)))
        self.params = res.params
        self.wald_chi = (res.params / res.bse) ** 2  # step backward 的 wald 检验统计量
        self.std_error = res.bse
        self.pchi2 = 2 * stats.norm.cdf(-np.abs((res.params / res.bse)))  # step backward 的 wald 检验 p 值

    def resprint(self):
        print("                          Model Fit Statistics ")
        print("==============================================================================")
        print("AIC     :         %s              BIC   :        %s    " % (self.aic, self.bic))
        print("-2Logl  :         %s              SC    :        %s    " % (self.logl, self.sc))
        print("==============================================================================")


class Checkio(object):
    """此类只打印 step in/out 过程的相关信息，不做其他实质性事情"""

    def __init__(self, xwait, score, pvalue):
        self.xwait = xwait
        self.score = score
        self.pvalue = pvalue

    def print_enter(self):
        print("              Analysis of Variables Eligible for Entry  ")
        print("==============================================================================")
        print("\t%5s\t \t%5s\t \t%5s\t" % ("variable", "Wald Chi-square", "Pr>ChiSq"))
        for i, v in enumerate(self.xwait):
            print("    \t%5s\t             \t%10s\t     \t%10s\t" % (v, self.score[i], self.pvalue[i]))
        print(" ")

    def print_remove(self):
        print("              Analysis of Variables Eligible for Remove  ")
        print("==============================================================================")
        print("\t%5s\t \t%5s\t \t%5s\t" % ("variable", "Wald Chi-square", "Pr>ChiSq"))
        for i, v in enumerate(self.xwait):
            print("    \t%5s\t             \t%10s\t     \t%10s\t" % (v, self.score[i], self.pvalue[i]))
        print(" ")


class GlobalNullTest(object):
    def __init__(self, x, y, beta):
        self.x = x
        self.p = (x.shape[1] - 1)
        self.y = np.array(y).reshape(len(y), 1)
        self.betai = pd.DataFrame(beta + [0.])

    def score(self):
        pi_value = 1 / (1 + np.exp(-1 * np.dot(self.x, self.betai)))
        #        print(self.x.shape,type(self.x))
        #        print(self.y.shape,type(self.y))
        #        print(pi_value.shape,type(pi_value))
        u = np.dot(self.x.T, self.y - pi_value)
        h = np.dot(self.x.T * (pi_value * (1 - pi_value)).reshape(len(pi_value)), self.x)
        score = np.dot(np.dot(u.T, np.linalg.inv(h)), u)
        #        return list(score[0])
        return score[0]  # 计算卡方统计量

    def pvalue(self):
        pvalue = stats.distributions.chi2.sf(self.score(), 1)
        return pvalue  # 卡方检验 p 值，step forward 需要用此检验


# return list(pvalue)

class StepwiseModel(object):
    __doc__ = """
    The Logistic Regression Model.

    Parameters
    -------------
    y: array-like
        The dependent variable, dim = n*1
    X: array-like
        The independnet variable, dim = n*p. By default, an intercept is included.
    weight: array-like
        Each observation in the input data set is weighted by the value of the WEIGHT variable. By default, weight is np.ones(n)
    method: ['forward', 'backward', 'stepwise']
        The default selection method is 'stepwise',in fact,only support stepwise now
    maxiter: int
        maxiter = 25 (default)
    mindiff: float
        mindiff = 1e-8 (default)

    Results
    -------------
    params: array
        Parameters' Estimates
    AIC: float
        Akaike information criterion.  `-2*(llf - p)` where `p` is the number
        of regressors including the intercept.
    BIC: float
        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
        number of regressors including the intercept.
    SC: float
        Schwarz criterion. `-LogL + p*(log(nobs))`
    std_error: Array
        The standard errors of the coefficients.(bse)
    Chi_Square: float
        Wald Chi-square : (logit_res.params[0]/logit_res.bse[0])**2
    Chisqprob: float
        P-value from Chi_square test statistic
    llf: float
        Value of the loglikelihood, as (LogL)
    Notes
    ----
    """

    def __init__(self, X, y, **kwargs):
        """ X为(n,p)DataFrame, y"""
        self.X = sm.add_constant(X)  # by default, include intercept in model
        self.y = y
        self.maxiter = 25  # default maxiter
        self.mindiff = 1e-8  # default mindiff
        self.method = 'None'
        self.slentry = 0.05  # default
        self.slstay = 0.05  # default
        self.weight = np.ones(len(y)) / len(y)  # 默认权重全为 1
        if 'slentry' in kwargs.keys():
            self.slentry = float(kwargs['slentry'])
        if 'slstay' in kwargs.keys():
            self.slstay = float(kwargs['slstay'])
        if 'method' in kwargs.keys():
            self.method = kwargs['method']
        if 'maxiter' in kwargs.keys():
            self.maxiter = kwargs['maxiter']
        if 'mindiff' in kwargs.keys():
            self.mindiff = kwargs['mindiff']
        if 'weight' in kwargs.keys():
            self.weight = kwargs['weight']
            self.weight /= self.weight.sum()

    def stepwise(self):
        n, p = self.X.shape[0], self.X.shape[1]
        #        y = pd.DataFrame(self.ydata()[0], columns = ['y'])
        xcol = list(self.X.columns)
        # xin  = np.ones(p)
        # xout = np.zeros(p)
        xenter = ['const']
        xwait = xcol.copy()
        xwait.remove('const')
        # xout   = []
        step = 0
        print("\n****** The LogitReg Process ******\n** Step 0. Intercept entered:")
        logit_mod = sm.Logit(self.y, self.X['const'])
        try:
            logit_res = logit_mod.fit(disp=0)
        except:
            logit_res = logit_mod.fit(disp=0, method='bfgs')
        #logit_res = logit_mod.fit(disp=0)
        Beta0 = list(logit_res.params)
        print(logit_res.summary2(), '\n')
        history = {}
        history['const'] = LRStats(step, n, 1, logit_res)
        history['const'].resprint()
        newx = self.X['const']
        for i in np.arange(p):
            print("   ")
            score = []
            pvalue = []
            rb = 0
            logit_res = {}
            history = {}
            for xname in xwait:
                _tmpxenter = xenter + [xname]
                _tmpx = self.X[_tmpxenter]
                logit_mod = sm.Logit(self.y, _tmpx)
                try:
                    logit_res[xname] = logit_mod.fit(disp=0)
                except:
                    logit_res[xname] = logit_mod.fit(disp=0, method='bfgs')
                history[xname] = LRStats(step, n, 1, logit_res[xname])
                nulltest = GlobalNullTest(_tmpx, self.y, Beta0)
                score.append(nulltest.score())
                pvalue.append(nulltest.pvalue())
            Checkio(xwait, score, pvalue).print_enter()  # 打印运行信息
            if (min(pvalue) <= self.slentry):
                # Update newx and xenter
                xin = [xwait[ii] for ii, pv in enumerate(pvalue) if pv == min(pvalue)][0]  # 最显著的变量选进来
                xenter.append(xin)
                newx = self.X[xenter]
                xwait.remove(xin)
                step += 1
                print("** step %s: %s entered:\n" % (step, xin))
                print(logit_res[xin].summary2())
                Beta0 = list(logit_res[xin].params)
                history[xin].resprint()
                pouttest = history[xin].pchi2[1:]
                waldouttest = history[xin].wald_chi[1:]  # step backward 剔除变量检验
                xouttest = xenter[1:]
                Checkio(xouttest, waldouttest, pouttest).print_remove()
                while 1:
                    if (max(pouttest) <= self.slstay):  # 不显著 就 不剔除
                        print(
                            "         No (additional) Variables met the %s significance level for remove into the model" % (
                                self.slstay))
                        break
                    else:
                        _slrindex = pouttest.argmax()
                        xout = xouttest[_slrindex]
                        step += 1
                        print("step %s: %s removed:\n" % (step, xout))
                        # Update newx and xenter
                        # print(xenter)
                        # print(newx)
                        del newx[xout]
                        xenter.remove(xout)
                        # xwait.remove(xout)
                        logit_mod = sm.Logit(self.y, newx)
                        try:
                            _logit_res = logit_mod.fit(disp=0)
                        except:
                            _logit_res = logit_mod.fit(disp=0, method='bfgs')
                        #_logit_res = logit_mod.fit(disp=0)
                        Beta0 = list(_logit_res.params)
                        _logit_res.summary2()
                        _history = LRStats(step, n, 1, _logit_res)
                        _history.resprint()
                        pouttest = _history.pchi2[1:]
                        waldouttest = _history.wald_chi[1:]
                        xouttest = xenter[1:]
                        Checkio(xouttest, waldouttest, pouttest).print_remove()
                        ij = 0
                        if (xin == xout and ij == 0):
                            print(
                                "stepwise terminates because the last effect enter "
                                "is removed by the Wald statistic criterion")
                            rb = 1
                            break
                        else:
                            ij += 1
                            rb = 2
            else:
                print("    No additional Variables met the %s significance level for entry into the model" % (
                    self.slentry))
                break
            if rb == 1:
                break
            newx = newx.T
            i += 1
        result = {}
        for iii, b in enumerate(Beta0):
            result[xenter[iii]] = b
        return result


class LR(ModelPlot):
    def __init__(self, features, target, df=None, train_df=None, valid_df=None, test_df=None):
        """
        :param features:  a list of feature
        :param target:  a sting
        :param df:
        :param train_df:
        :param valid_df:
        :param test_df:
        """
        if train_df and valid_df and test_df:
            self.train_df = train_df
            self.valid_df = valid_df
            self.test_df = test_df
        elif df:
            train_valid_dict = split_df(df, pct=0.4)
            self.train_df = train_valid_dict['train']
            valid_test_dict = split_df(train_valid_dict['test'], pct=0.5)
            self.valid_df = valid_test_dict['train']
            self.test_df = valid_test_dict['test']
        else:
            print('please input data')
        self.features = features
        self.target = target

    def train(self):
        self.model = logit_fit(self.train_df, self.target, self.features)
        train_plot = ModelPlot(self.model[1], y_variable=self.target, title='Train')
        train_plot.roc_plot()
        train_plot.ks_plot()

    def valid(self):
        predit_data = logit_predict(self.valid_df, self.target, self.features, self.model[0])
        valid_plot = ModelPlot(predit_data, y_variable=self.target, title='Valid')
        valid_plot.roc_plot()
        valid_plot.ks_plot()

    def refit(self):
        model_v = logit_fit(self.valid_df, self.target, self.features)

    def test(self):
        predit_data_t = logit_predict(self.test_df, self.target, self.features, self.model[0])
        test_plot = ModelPlot(predit_data_t, y_variable=self.target, title='Test')
        test_plot.roc_plot()
        test_plot.ks_plot()


"""
下面是将分箱后的表转换成可map的ifelse
"""


def create_if(r, varname=None, target=None):
    left = str(r['left_value'])
    right = str(r['right_value'])
    value = str(r[target])
    variable = str(r['var'].split('_bin')[0])
    left_exp = "if r['" + variable + "'] > " + left
    elif_exp = "elif r['" + variable + "'] > " + left
    right_exp = " and r['" + variable + "'] <= " + right + ":\n"
    return_value = 2 * indent + "return " + value
    if r['left_value'] == -999999999:
        return indent + left_exp + right_exp + return_value
    else:
        return indent + elif_exp + right_exp + return_value


def add_def(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("def {}_map(r):".format(kwargs.get('varname')))
        return func(*args, **kwargs)

    return wrapper


@add_def
def parse_var(df, varname=None, target=None):
    min_value = df['left_value'].min()
    max_value = df['right_value'].max()
    df = df.sort_values(by='left_value')
    df['left_value'] = df['left_value'].replace({min_value: -999999999})
    df['right_value'] = df['right_value'].replace({max_value: 999999999})
    df['result'] = df.apply(create_if, varname=varname, target=target, axis=1)
    for i in df['result']:
        print(i)
    print(indent + "else:\n" + 2 * indent + "return 'error'")


def parse_min(r):
    split_list = r.split('~')
    if len(split_list) == 2:
        return float(split_list[0])
    else:
        return 'error'


def parse_max(r):
    split_list = r.split('~')
    if len(split_list) == 2:
        return float(split_list[1])
    else:
        return 'error'


def deploy_lr_file(file, func_file_name, target=None):
    if file.endswith('pkl'):
        df = pd.read_pickle(file)
    elif file.endswith('xlsx'):
        df = pd.read_excel(file)
    elif file.endswith('xls'):
        df = pd.read_excel(file)
    elif file.endswith('csv'):
        df = pd.read_csv(file)
    else:
        print('不支持的文件类型')
    df = df.loc[df.bin_group.notnull()]
    df = df.loc[~(df.bin_group.isin(['inf~inf', '-inf~inf']))]
    df = df.loc[df.bin_group.str.contains('~', na=False)]
    variables = df['var'].unique().tolist()
    df['bin_group'] = df['bin_group'].str.replace('inf', '9999999999999')
    df['left_value'] = df['bin_group'].map(parse_min)
    df['right_value'] = df['bin_group'].map(parse_max)
    orig_stdout = sys.stdout
    f = open(func_file_name, 'w')
    sys.stdout = f
    for i in variables:
        data = df.loc[df['var'] == i]
        varname = i.split('_')[0]
        parse_var(data, varname=varname, target=target)
    sys.stdout = orig_stdout
    f.close()


def deploy_lr_df(stats_df, func_file_name, target=None):
    df = stats_df.loc[stats_df.bin_group.notnull()]
    
    df = df.loc[~(df.bin_group.isin(['inf~inf', '-inf~inf']))]
    df = df.loc[df.bin_group.str.contains('~', na=False)]
    variables = df['var'].unique().tolist()
    df['bin_group'] = df['bin_group'].str.replace('inf', '9999999999999')
    df['left_value'] = df['bin_group'].map(parse_min)
    df['right_value'] = df['bin_group'].map(parse_max)
    orig_stdout = sys.stdout
    f = open(func_file_name, 'w')
    sys.stdout = f
    for i in variables:
        data = df.loc[df['var'] == i]
        varname = i.split('_bin')[0]
        parse_var(data, varname=varname, target=target)
    sys.stdout = orig_stdout
    f.close()


class ScoreMeasure(object):
    """
       beta_dict = {'const' : -1.670622,
            'id276_bin' :  -0.742585,
            'id322_bin' :  -0.757020,
            'id366_bin' : -0.786014,
            'id267_bin' : -0.493884,
            'id310_bin' : -0.660974,
            'id307_bin' : -0.691831}
         """
    def __init__(self,beta_dict,p0=600,pdo=40,theta0=0.08):
        self.factor = pdo/np.log(2)
        self.offset = p0+self.factor*np.log(theta0)
        self.beta_dict = beta_dict
    
    def get_basescore(self):
        return round(self.offset-self.beta_dict.get('const')*self.factor,4)
    
    def add_stats_df(self,stats_df):
        stats_df_1 = stats_df.copy()
        stats_df_1['beta'] = stats_df_1['var'].map(self.beta_dict)
        stats_df_1['score'] = -self.factor*stats_df_1['beta']*stats_df_1['woe']
        stats_df_1['score'] = stats_df_1['score'].map(lambda x: round(x,4))
        return stats_df_1
        
        

    
    
