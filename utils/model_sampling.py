import json
from random import sample, seed
import pandas as pd
from models.pytorch_cvae import generate_group_samples, generate_samples_with_allocation, house_sampling, load_model_and_dataset, read_samples, save_samples, train_torch_cvae
import pandasql
import numpy as np
import re
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType,LongType
import time
import torch
from keras.utils.np_utils import to_categorical
import logging
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

logger = logging.getLogger(__name__)

def lowercase(query_table):
    query_table.database_name=query_table.database_name.lower()
    query_table.table_name=query_table.table_name.lower()
    query_table.qualified_table_name=query_table.qualified_table_name.lower()
    query_table.join_col=query_table.join_col.lower()
    query_table.columns=[t.lower() for t in query_table.columns]
    query_table.involved_cols=[t.lower() for t in query_table.involved_cols]
    query_table.sum_cols=[t.lower() for t in query_table.sum_cols]
    query_table.avg_cols=[t.lower() for t in query_table.avg_cols]
    query_table.agg_cols=[t.lower() for t in query_table.agg_cols]
    query_table.group_bys=[t.lower() for t in query_table.group_bys]
    query_table.group_cols=[t.lower() for t in query_table.group_cols]
    query_table.config_file=query_table.config_file.lower()

def query_multi_sampling(query,results,training_times):
    result,training_time = query_sampling(query)
    results.append(result)
    training_times.append(training_time)
    return results,training_times

def query_sampling(query):
    query_tables = query.query_tables
    samples = {}
    training_time=0.0
    for query_table in query_tables:
        lowercase(query_table)
        with open(query_table.config_file) as f:
            train_config = json.load(f)
            logger.info("load train config {} successfully".format(query_table.config_file))
        train_start=time.perf_counter()
        model,dataset=load_model_and_dataset(train_config)
        if model is None or dataset is None:
            model,dataset = train_torch_cvae(train_config)
        training_time+=time.perf_counter()-train_start
        sample=generate_model_samples(model, dataset,query_table, train_config)
        samples[query_table.table_name] = sample
    start_time=time.perf_counter()
    
    # sql = pandasql.PandaSQL()
    # rewrited_sql = rewrite_sql(query)
    # logger.info('rewrited sql:'+rewrited_sql)
    # res = sql(rewrited_sql, env=samples)
    # new_cols = [re.sub(r"^(SUM|AVG)\_([\w]+)", r"\1(\2)", t)
    #             for t in res.columns]
    # res.columns = new_cols
    res=sample_aggregation(samples,query)
    logger.info("execute rewrited query time:{}".format(
            (time.perf_counter() - start_time)))
    return res,training_time


def rewrite_sql(query):
    rewrited_sql = query.sql.lower()
    query_tables = query.query_tables
    num_cols = []
    rate_cols = []
    outlier_flag = False

    for qt in query_tables:
        if len(qt.group_num_col) > 0:
            num_cols.append(qt.group_num_col)
        if len(qt.rate_col) > 0:
            rate_cols.append(qt.rate_col)
        outlier_flag |= qt.outlier_flag
        rewrited_sql = rewrited_sql.replace(
            qt.qualified_table_name, qt.table_name)
        for condition in qt.conditions:
            rewrited_sql = rewrited_sql.replace(condition, "1=1")

    if len(num_cols) > 0:
        # rewrited_sql = re.sub("SUM\\((.*?)\\)", r"SUM(\1/(" + "*".join(
        #     rate_cols) + ")*(" + "*".join(num_cols) + r")) AS SUM_\1", rewrited_sql)
        rewrited_sql = re.sub("sum\\((.*?)\\)", r"SUM(\1*(" + "*".join(num_cols) + r")) AS SUM_\1", rewrited_sql)
    else:
        rewrited_sql = re.sub(
            "sum\\((.*?)\\)", r"SUM(\1/(" + "*".join(rate_cols) + r")) AS SUM_\1", rewrited_sql)
        # rewrited_sql = re.sub("SUM\\((.*?)\\)", r"SUM(\1*(" + "*".join(num_cols) + r")) AS SUM_\1", rewrited_sql)
    

    # splits=rewrited_sql.split("FROM")
    # rewrited_sql=splits[0]+",COUNT(*) AS SAMPLE_NUM FROM"+splits[1]
    
    if outlier_flag:
        if len(num_cols) > 0:
            rewrited_sql = re.sub("avg\\((.*?)\\)", r"SUM(\1*(" + "*".join(
                num_cols) + "))/SUM(" + "*".join(num_cols) + r") AS AVG_\1", rewrited_sql)
        else:
            rewrited_sql = re.sub("avg\\((.*?)\\)", r"SUM(\1/(" + "*".join(
                rate_cols) + "))/SUM(1/(" + "*".join(rate_cols) + r")) AS AVG_\1", rewrited_sql)
    
    return rewrited_sql

def sample_aggregation(sample_dict, query):
    sample_list=[t for t in sample_dict.values()]
    sum_cols=[col for qt in query.query_tables for col in qt.sum_cols]
    avg_cols=[col for qt in query.query_tables for col in qt.avg_cols]
    join_cols = [qt.join_col for qt in query.query_tables]
    groupby_cols = [t.attr_name.lower() for t in query.group_bys]
    rate_cols = [qt.rate_col for qt in query.query_tables]
    outlier = False
    for qt in query.query_tables:
        outlier |= qt.outlier_flag
    aggregations = {}

    for col in avg_cols:
        # agg_name = col + "_mean"
        agg_name = "avg({})".format(col)
        aggregations[agg_name] = (col, 'mean')

    for col in sum_cols:
        # agg_name = col + "_sum"
        agg_name = "sum({})".format(col)
        aggregations[agg_name] = (col, 'sum')

    for col in rate_cols:
        agg_name = col
        aggregations[agg_name] = (col, 'mean')

    if len(sample_list) > 1:
        sample_table = pd.merge(sample_list[0], sample_list[1], left_on=join_cols[0], right_on=join_cols[1], how='inner')
    else:
        sample_table=sample_list[0]

    if len(groupby_cols) > 0:  # with group by clause
        if not outlier:  ### no outliers
            agg_result = sample_table.groupby(by=groupby_cols).agg(**aggregations)
            rate_col = 'rate'
            agg_result[rate_col] = 1
            for col in rate_cols:
                agg_result[rate_col] *= agg_result[col]
                del agg_result[col]
            for col in agg_result.columns:
                if col.startswith('sum'):
                    # if col.endswith('_sum'):
                    agg_result[col] /= agg_result[rate_col]
            del agg_result[rate_col]
        else:  ### with outliers
            logger.info("outliers aggregation")
            cnt_col = 'cnt'
            aggregations[cnt_col] = (avg_cols[0], 'size')
            agg_result = sample_table.groupby(by=groupby_cols + rate_cols).agg(**aggregations)
            # print(agg_result)
            rate_col = 'rate'
            agg_result[rate_col] = 1
            for col in rate_cols:
                agg_result[rate_col] *= agg_result[col]
                del agg_result[col]
            f_aggregations = {}
            for col in agg_result.columns:
                if col.startswith('sum') or col == 'cnt':
                    # if col.endswith('_sum'):
                    agg_result[col] /= agg_result[rate_col]
                    f_aggregations[col] = (col, 'sum')
            agg_result = agg_result.groupby(by=groupby_cols).agg(**f_aggregations)
            for col in avg_cols:
                # agg_name = col + "_mean"
                magg_name = "avg({})".format(col)
                sagg_name = "sum({})".format(col)
                agg_result[magg_name] = agg_result[sagg_name] / agg_result[cnt_col]
            del agg_result[cnt_col]
    else:  # without group by clause
            cnt_col = 'cnt'
            aggregations = {agg_name: aggregations[agg_name] for agg_name in aggregations if agg_name.startswith('sum')}
            aggregations[cnt_col] = (avg_cols[0], 'size')
            agg_result = sample_table.groupby(by=rate_cols).agg(**aggregations)
            agg_result.reset_index(inplace=True)
            # print(agg_result)
            rate_col = 'rate'
            agg_result[rate_col] = 1
            for col in agg_result.columns:
                if col.endswith('_rate'):
                    agg_result[rate_col] *= agg_result[col]
                    del agg_result[col]
            for col in agg_result.columns:
                if col.startswith('sum') or col == 'cnt':
                    agg_result[col] /= agg_result[rate_col]
            del agg_result[rate_col]
            agg_result = pd.DataFrame(agg_result.sum()).transpose()
            # agg_result = agg_result.agg(**f_aggregations)
            for col in avg_cols:
                # agg_name = col + "_mean"
                magg_name = "avg({})".format(col)
                sagg_name = "sum({})".format(col)
                agg_result[magg_name] = agg_result[sagg_name] / agg_result[cnt_col]
            del agg_result[cnt_col]
    return agg_result

def house_sampling(model, dataset, sample_rate):
    logger.info("house sampling rate:{}".format(sample_rate))
    model.eval()
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    sample_rates = {}
    sample_allocation = {}
    
    for label_value_idx, label_value in label_value_mapping.items():
        label_count = label_group_counts[label_value]
        sample_count = round(label_count * sample_rate)
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / label_count
    return sample_allocation, sample_rates


def senate_sampling(model, dataset, sample_rate):
    model.eval()
    logger.info("senate sampling rate:{}".format(sample_rate))
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate
    group_nums = len(label_group_counts)
    each_group_samples = int(total_samples / group_nums) + 1
   
    sample_allocation = {}
    sample_rates = {}
    for label_value_idx, label_value in label_value_mapping.items():
        label_count = label_group_counts[label_value]
        sample_count = each_group_samples if each_group_samples < label_count else label_count
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / label_count
    return sample_allocation, sample_rates

def statistics_sampling(model, dataset, sample_rate, query_table):
    model.eval()
    logger.info("statistics sampling rate:{}".format(sample_rate))
    numeric_columns = list(set(query_table.sum_cols) & set(query_table.avg_cols) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate
    sample_allocation = {}
    sample_rates = {}
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        relative_variances = sum([label_group_relative_stds[col][label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(total_samples * (relative_variances / sum_relative_variance))
        sample_count = group_sample if group_sample < group_count else group_count
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / group_count
    # logger.info("statistics sampling allocation:{}".format(sample_allocation))
    return sample_allocation, sample_rates

def statistics_sampling_with_small_group(model, dataset, sample_rate, query_table):
    model.eval()
    logger.info("statistics with small group sampling rate:{}".format(sample_rate))
    numeric_columns = list(set(query_table.sum_cols) & set(query_table.avg_cols) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate

    statistics_sampling_samples = total_samples * 0.5
    small_group_sampling_samples = total_samples - statistics_sampling_samples
    small_group_K = small_group_sampling_samples / len(label_group_counts)

    sample_allocation = {}
    sample_rates = {}
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        relative_variances = sum([label_group_relative_stds[col][label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(statistics_sampling_samples * (relative_variances / sum_relative_variance))
        group_sample += small_group_K
        sample_count = group_sample if group_sample < group_count else group_count
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / group_count

    return sample_allocation, sample_rates


def generate_model_samples(model, dataset, query_table, train_config):
    # sample_rate = train_config["sample_rate"]
    sample_rate=query_table.sampling_rate
    if train_config['sample_method'] == "senate":
        sample_allocation, sample_rates = senate_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "house":
        sample_allocation, sample_rates = house_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "statistics":
        sample_allocation, sample_rates = statistics_sampling(model, dataset, sample_rate,
                                                                               query_table)
    else:
        sample_allocation, sample_rates = statistics_sampling_with_small_group(model, dataset, sample_rate,
                                                                               query_table)
    # print(sample_allocation)
    if len(query_table.conditions)>0:
        logger.info("filtering with condition {}".format(query_table.conditions))
        condition = query_table.conditions[0]
        if '<=' in condition:
            bound_value = int(condition.split('<=')[-1])
            sample_allocation = {k: v for k, v in sample_allocation.items() if k <= bound_value}
        elif '>=' in condition:
            bound_value = int(condition.split('>=')[-1])
            sample_allocation = {k: v for k, v in sample_allocation.items() if k >= bound_value}
        elif '=' in condition:
            bound_value = int(condition.split('=')[-1])
            sample_allocation = {k: v for k, v in sample_allocation.items() if k == bound_value}
        elif '<' in condition:
            bound_value = int(condition.split('<')[-1])
            sample_allocation = {k: v for k, v in sample_allocation.items() if k < bound_value}
        elif '>' in condition:
            bound_value = int(condition.split('>')[-1])
            sample_allocation = {k: v for k, v in sample_allocation.items() if k > bound_value}

    samples = generate_model_samples_with_allocation(dataset, model, sample_allocation, train_config)
    rate_col = query_table.table_name+"_rate"
    query_table.rate_col = rate_col
    samples[rate_col] = samples[dataset.label_column_name].map(sample_rates)

    # group_num_col = query_table.table_name+"_GROUP_NUM"
    # query_table.group_num_col = group_num_col
    # aggs={}
    # for col_name in query_table.agg_cols:
    #     aggs[col_name] = (col_name, 'mean')
    # agg_table=samples.groupby(query_table.group_cols+[dataset.label_column_name]).agg(**aggs).reset_index()
    # # agg_table["GROUP_ID"]=agg_table[query_table.group_cols].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
    # # agg_table["GROUP_ID"] = agg_table[query_table.group_cols].astype(str).agg('-'.join, axis=1)
    # group_info={str(k):v for k,v in dataset.label_group_counts.items()}
    # agg_table[group_num_col]=agg_table[dataset.label_column_name].apply(lambda x: group_info[str(x)] if str(x) in group_info else 1)
    # samples = agg_table.drop(dataset.label_column_name, 1)
    # print(samples[:10])
    if 'outliers' in train_config and train_config['outliers'] == 'true':
        logger.info("outlier appending")
        outliers=dataset.outliers
        # outliers[group_num_col]=1
        outliers[rate_col]=1.0
        samples = pd.concat([samples, outliers])
        query_table.outlier_flag=True
    return samples

def generate_model_samples_with_allocation(dataset, model, sample_allocation,
                                     train_config):
    start_time = time.perf_counter()
    batch_size = train_config["batch_size"]
    latent_dim = train_config["latent_dim"]
    categorical_encoding = train_config["categorical_encoding"]
    z_decoded = []
    label_value_mapping = dataset.label_value_mapping
    label_size = len(label_value_mapping)
    threads=[]

    for label_value_idx, label_value in label_value_mapping.items():
        if label_value in sample_allocation:
            sample_count = int(sample_allocation[label_value])
            if categorical_encoding == 'binary':
                mapping = dataset.label_mapping_out
                label = [mapping.loc[label_value_idx].values]
                label = torch.from_numpy(np.repeat(label, batch_size, axis=0)).to(model.device)
                # label = np.tile(label, (sample_count, 1))
            else:
                label = np.ones((batch_size,)) * label_value_idx
                label = torch.from_numpy(to_categorical(label, label_size)).to(model.device)
            generate_group_samples(sample_count, label, latent_dim, batch_size, model, z_decoded)

    z_decoded = np.concatenate(z_decoded, axis=0)
    samples_df = dataset.decode_samples(z_decoded)
    # samples_df['{}_rate'.format(dataset.name)] = samples_df[dataset.label_column_name].map(sample_rates)
    end_time = time.perf_counter()
    logger.info('sampling time:{}'.format(end_time - start_time))
    return samples_df