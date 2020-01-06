"""
n-folds lgbm model parser
"""

import numpy as np
import json
import time
import os
from multiprocessing import Pool
from sklearn.externals.joblib import Parallel, delayed
from statsmodels.distributions import ECDF


def decision(row, threshold, default_left):
    if (np.isnan(row))and (default_left is True):
        return 'left_child'
    elif row <= threshold:
        return 'left_child'
    else:
        return 'right_child'


def tree_parser(row, model, feature_names):
    score = 0
    for i in range(len(model['tree_info'])):
        num_leaves = model['tree_info'][i]['num_leaves']
        tree = model['tree_info'][i]['tree_structure']
        for i in range(num_leaves):
            threshold = tree.get('threshold')
            default_left = tree.get('default_left')
            split_feature = feature_names[tree['split_feature']]
            next_decison = decision(
                row[split_feature], threshold, default_left)
            tree = tree[next_decison]
            if tree.get('left_child', 'not found') == 'not found':
                score = score + tree['leaf_value']
                break
    return score


def paralell_predict(input_args):
    data = input_args[0]
    model = input_args[1]
    feature_names = model['feature_names']
    predict = Parallel(
        n_jobs=2)(
        delayed(tree_parser)(
            v,
            model,
            feature_names) for k,
        v in data.iterrows())
    #predict = data.apply(lambda x: tree_parser(x, model, feature_names), axis=1).values
    return np.array(predict)


def transformer(json_path, test_df):
    start_time = time.time()
    model_file = json.load(open(json_path, "rb"))
    n = len(model_file)
    input_args = [
        [test_df, model_file['fold{0}_tree'.format(i)]] for i in range(1, n)]
    cpu_count = os.cpu_count()
    with Pool(cpu_count) as p:
        process_result = p.map(paralell_predict, input_args)

    predict_test = np.zeros(test_df.shape[0])
    for fold_predict in process_result:
        predict_test += (1 / (np.exp(-fold_predict) + 1)) / (n - 1)

    ecdf = ECDF(model_file['ecdf'])
    oof_test_ecdf = ecdf(predict_test)
    print("- Parser time consumed: %s seconds" % int(time.time() - start_time))
    return oof_test_ecdf
