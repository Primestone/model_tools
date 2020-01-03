# -*- coding: utf-8 -*-

"""
visual

@author: Moon
"""

import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
# color = sns.color_palette()
import plotly.offline as py
import cufflinks as cf
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot

offline.init_notebook_mode()
py.init_notebook_mode(connected=True)
cf.go_offline()
init_notebook_mode(connected=True)


def miss_dist(data):
    if data.shape[0] > 5000:
        data = data.iloc[:50000, :]

    des = data.describe().T
    des['miss_pct'] = 1 - des['count'] / data.shape[0]
    des = des.sort_values(by='miss_pct', ascending=False)

    if data.shape[1] < 50:
        show_cols = data.columns
    else:
        show_cols = des.index.tolist()[:50]

    msno.matrix(df=data[show_cols], figsize=(20, 14), color=(0.9, 0.2, 0.1))
    plt.show()


def cat_dist(data, column, color='green', kind='bar'):
    vc = data[column].value_counts(dropna=False)
    if vc.shape[0] > 16:
        rare_list = vc.index.tolist()[16:]
        data[column] = data[column].apply(lambda x: 'other' if x in rare_list else x)
        vc = data[column].value_counts(dropna=False)

    if kind == 'bar':
        vc.iplot(kind='bar', xTitle=column, yTitle="Count", title='{}\'s Distribution '.format(column), color=color)
    elif kind == 'pie':
        df = pd.DataFrame({'labels': vc.index,
                           'values': vc.values})
        df.iplot(kind='pie', labels='labels', values='values', title='{}\'s Distribution'.format(column), hole=0.5)


def cat_dist_with_target(data, column, target, barmode='stack'):
    temp = data[column].value_counts(dropna=False)
    temp_y0 = []
    temp_y1 = []
    for val in temp.index:
        temp_y1.append(np.sum(data[target][data[column] == val] == 1))
        temp_y0.append(np.sum(data[target][data[column] == val] == 0))
    trace1 = go.Bar(
        x=temp.index,
        y=(temp_y1 / temp.sum()) * 100,
        name='YES'
    )
    trace2 = go.Bar(
        x=temp.index,
        y=(temp_y0 / temp.sum()) * 100,
        name='NO'
    )

    data = [trace1, trace2]
    layout = go.Layout(
        title="{}'s distribution with TARGET in %".format(column),
        barmode=barmode,
        # width=1000,
        xaxis=dict(
            title='{}'.format(column),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Count in %',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def num_dist(data, column, smoth=1.0):
    x1 = data[column].fillna(0).values / smoth
    fig = ff.create_distplot([x1], group_labels=[column], bin_size=0.3)
    fig['layout'].update(title="{}'s Distribution".format(column))
    iplot(fig)


def num_dist_with_targte(data, column, target, smoth=1.0):
    x1 = data[data[target] == 1][column].fillna(0).values / smoth
    x0 = data[data[target] == 0][column].fillna(0).values / smoth
    fig = ff.create_distplot([x1, x0], group_labels=['Yes', 'No'], bin_size=0.3)
    fig['layout'].update(title="{}'s Distribution with TARGET".format(column))
    iplot(fig)


def heatmap(data, columns, cmp="YlGnBu"):
    plt.figure(figsize=(16, 12))
    plt.title("Variables Correction ")
    corr = data[columns].corr()
    sns.heatmap(corr, cmap=cmp)
    plt.show()
