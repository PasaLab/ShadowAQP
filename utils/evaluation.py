import pandas as pd
import numpy as np

# Press the green button in the gutter to run the script.
def compare_aggregation(sample_agg, ground_truth_path, index=False):
    if index:
        ground_truth = pd.read_csv(ground_truth_path, index_col=0)
    else:
        ground_truth = pd.read_csv(ground_truth_path)
    diff = (ground_truth - sample_agg).abs() / ground_truth
    diff.fillna(1, inplace=True)
    # print("aqp result",sample_agg)
    # print("ground truth:",ground_truth)
    return diff


def compare_aggregation_norm(sample_agg, ground_truth_path, index=False):
    if index:
        ground_truth = pd.read_csv(ground_truth_path, index_col=0)
    else:
        ground_truth = pd.read_csv(ground_truth_path)
    diff = 1 - np.exp(-((ground_truth - sample_agg).abs() / ground_truth))
    diff.fillna(1, inplace=True)
    # print(sample_agg)
    # print(ground_truth)
    return diff


def compare_aggregation_norm_var(sample_agg, query_config, index=False):
    ground_truth_path = query_config['ground_truth']
    var_path = query_config['var']
    if index:
        ground_truth = pd.read_csv(ground_truth_path, index_col=0)
        var = pd.read_csv(var_path, index_col=0)
    else:
        ground_truth = pd.read_csv(ground_truth_path)
        var = pd.read_csv(var_path)
    diff = 1 - np.exp(-((ground_truth - sample_agg) ** 2).div(var))
    diff.fillna(1, inplace=True)
    # print(sample_agg)
    # print(ground_truth)
    return diff