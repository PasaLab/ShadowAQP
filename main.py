import argparse
import logging
import json
import os
import sys
import threading
# from models.keras_vae import train_vae
# from models.keras_cvae import train_cvae
import time

from models.pytorch_cvae import train_torch_cvae, load_model_and_dataset, load_model_and_dataset_retrain, generate_samples
from utils.dataset_utils import TabularDataset, save_dataset, load_dataset
import pandas as pd
import numpy as np
from utils.plot_utils import plot

from pyspark.sql import SparkSession

os.environ['NUMEXPR_MAX_THREADS'] = '16'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', -1)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
#                     # filename='./skew_size_var/logs/aggvar086_id200_ld_200.log',
#                     filemode='w',
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#                     )
logger = logging.getLogger(__name__)
spark = SparkSession.builder.appName("SparkSQLSampling").master("yarn").config("spark.executor.memory", "20g").config(
    "spark.driver.memory", "30g").config("spark.executor.instances", 10).config("spark.executor.cores", 10).enableHiveSupport().getOrCreate()


def print_param(param):
    logger.info("model:{}".format(param["model_type"]))
    logger.info("batch size:{}".format(param["batch_size"]))
    logger.info("categorical columns:{}".format(param["categorical_columns"]))
    logger.info("numeric columns:{}".format(param["numeric_columns"]))
    if 'label_columns' in param:
        logger.info("label columns:{}".format(param["label_columns"]))
    logger.info("categorical encoding:{}".format(param["categorical_encoding"]))
    logger.info("numeric encoding:{}".format(param["numeric_encoding"]))


def train_load_models(train_config_list):
    model_dataset_list = []
    for param in train_config_list:
        model_type = param["model_type"]
        if model_type == "torch_cvae":
            if param['train_flag'] == 'train':
                model_dataset = train_torch_cvae(param)
            elif param['train_flag'] == 'load':
                model_dataset = load_model_and_dataset(param)
            else:
                model_dataset = load_model_and_dataset_retrain(param)
        model_dataset_list.append(model_dataset)
    return model_dataset_list


def train_models(train_config_list):
    model_dataset_list = []
    for param in train_config_list:
        model_type = param["model_type"]
        if model_type == "torch_cvae":
            model_dataset = train_torch_cvae(param)
        model_dataset_list.append(model_dataset)
    return model_dataset_list


def load_models(train_config_list):
    model_dataset_list = []
    for param in train_config_list:
        model_type = param["model_type"]
        if model_type == "torch_cvae":
            model_dataset = load_model_and_dataset(param)
        model_dataset_list.append(model_dataset)
    return model_dataset_list


def generate_sample_list(model_dataset_list, query_config, train_config_list):
    sample_list = []
    for i in range(len(train_config_list)):
        model, dataset = model_dataset_list[i]
        # sample = generate_samples(model, dataset, query_config, train_config_list[i])
        if train_config_list[i]['operation'] == 'aqp':
            sample = generate_samples(model, dataset, query_config, train_config_list[i])
        else:
            full_dataset = load_dataset(train_config_list[i])
            sample = full_dataset.origin_df
            sample['{}_rate'.format(dataset.name)] = 1.0
        # sample.to_csv('./samples/' + train_config_list[i]['name'] + '.csv', index=False)
        # if train_config_list[i]['name'].endswith('store'):
        #     print(sample)
        sample_list.append(sample)
    return sample_list


def compare_aggregation(sample_agg, query_config, index=False):
    ground_truth_path = query_config['ground_truth']
    if index:
        ground_truth = pd.read_csv(ground_truth_path, index_col=query_config['groupby_cols'])
    else:
        ground_truth = pd.read_csv(ground_truth_path)
    diff = ((ground_truth - sample_agg).abs() / ground_truth.abs())

    logger.info("aqp result:\n{}".format(sample_agg[:50]))
    logger.info("ground truth:\n{}".format(ground_truth[:50]))
    diff.fillna(1, inplace=True)
    return diff


def compare_aggregation_norm(sample_agg, query_config, index=False):
    ground_truth_path = query_config['ground_truth']
    if index:
        ground_truth = pd.read_csv(ground_truth_path, index_col=query_config['groupby_cols'])
    else:
        ground_truth = pd.read_csv(ground_truth_path)
    diff = 1 - np.exp(-((ground_truth - sample_agg).abs() / ground_truth.abs()))
    diff.fillna(1, inplace=True)
    return diff


def uniform_aqp(query_config, train_config_list):
    start_time = time.perf_counter()
    sample_list = uniform_sample_list(train_config_list)
    # sample_agg = ground_truth_aggregation(sample_list, query_config)

    sample_agg = sample_aggregation(sample_list, query_config, train_config_list)

    diff = compare_aggregation(sample_agg, query_config)
    logger.info("uniform aqp diff:\n{}".format(diff[:50]))
    logger.info("total error:{}".format(diff.values.sum() / diff.size))
    end_time = time.perf_counter()
    logger.info("sample time:{}".format(end_time - start_time))


def stratified_aqp(query_config, train_config_list):
    start_time = time.perf_counter()
    sample_list = uniform_sample_list(train_config_list)
    # sample_agg = ground_truth_aggregation(sample_list, query_config)
    sample_agg = sample_aggregation(sample_list, query_config, train_config_list)
    diff = compare_aggregation(sample_agg, query_config)
    logger.info("stratified aqp diff:\n{}".format(diff[:50]))
    logger.info("total error:{}".format(diff.values.sum() / diff.size))
    end_time = time.perf_counter()
    logger.info("sample time:{}".format(end_time - start_time))


def uniform_sample_list(train_config_list):
    sample_list = []
    for param in train_config_list:
        file_path = param["data"]
        delimiter = param["delimiter"]
        data = pd.read_csv(file_path, delimiter=delimiter)
        rate = param["sample_rate"]
        sample = data.sample(frac=rate)
        rate_col = '{}_rate'.format(param["name"])
        sample[rate_col] = rate
        sample_list.append(sample)
    return sample_list


def stratified_allocation(df, groupby_col, sample_rate, type):
    groupby_cnt = df.groupby(groupby_col).size()
    if type == 'house':
        allocation = (groupby_cnt * sample_rate).astype(int)
    else:
        k = int(groupby_cnt.sum() * sample_rate / groupby_col.size)
        allocation = groupby_cnt.apply(lambda x: x if x < k else k)
    allocation = allocation.to_dict()
    return allocation


def stratified_sample_list(train_config_list):
    sample_list = []
    for param in train_config_list:
        file_path = param["data"]
        delimiter = param["delimiter"]
        data = pd.read_csv(file_path, delimiter=delimiter)
        rate = param["sample_rate"]
        sample = data.sample(frac=rate)
        rate_col = '{}_rate'.format(param["name"])
        sample[rate_col] = rate
        sample_list.append(sample)
    return sample_list


def ground_truth_aggregation(query_config, train_config_list):
    start_time = time.perf_counter()
    sum_cols = query_config['sum_cols']
    avg_cols = query_config['avg_cols']
    join_cols = query_config['join_cols']
    groupby_cols = query_config['groupby_cols']
    ground_truth_path = query_config['ground_truth']

    aggregations = {}

    for col in avg_cols:
        # agg_name = col + "_mean"
        agg_name = "avg({})".format(col)
        aggregations[agg_name] = (col, 'mean')

    for col in sum_cols:
        # agg_name = col + "_sum"
        agg_name = "sum({})".format(col)
        aggregations[agg_name] = (col, 'sum')

    data_list = []
    for param in train_config_list:
        file_path = param["data"]
        delimiter = param["delimiter"]
        data = pd.read_csv(file_path, delimiter=delimiter)
        data_list.append(data)

    if len(data_list) > 1:
        join_result = pd.merge(data_list[0], data_list[1], left_on=join_cols[0], right_on=join_cols[1], how='inner')
        agg_result = join_result.groupby(by=groupby_cols).agg(**aggregations)
    else:
        samples = data_list[0]
        agg_result = samples.groupby(by=groupby_cols).agg(**aggregations)
    agg_result.to_csv(ground_truth_path)
    logger.info("ground truth result:\n{}".format(agg_result[:50]))
    end_time = time.perf_counter()
    logger.info("save ground truth to path:{}".format(ground_truth_path))
    logger.info("ground truth query time elapsed:{}".format(end_time - start_time))
    return agg_result


def sample_aggregation(sample_list, query_config, train_config_list, createDataFrame_time):
    sum_cols = query_config['sum_cols']
    avg_cols = query_config['avg_cols']
    join_cols = query_config['join_cols']
    groupby_cols = query_config['groupby_cols']
    rate_cols = [config['name'] + "_rate" for config in train_config_list]
    outlier = True if 'outliers' in train_config_list[0] and train_config_list[0]['outliers'] == 'true' else False
    aggregations = {}
    condition = 1
    if 'condition' in query_config and len(query_config['condition']):
        logger.info("filtering with condition {}".format(query_config['condition'][0]))
        condition = query_config['condition'][1]

    if len(sample_list) > 1:
        start_time = time.perf_counter()
        sample_list0 = spark.createDataFrame(sample_list[0], list(sample_list[0]))
        sample_list1 = spark.createDataFrame(sample_list[1], list(sample_list[1]))
        end_time = time.perf_counter()
        createDataFrame_time[0] = end_time - start_time
        logger.info("createDataFrame_time:{}".format(createDataFrame_time[0]))

        start_time = time.perf_counter()
        if query_config['name'] == 'census2_self_join':
            join_result = sample_list0.join(sample_list1, sample_list0.a_education_num==sample_list1.b_education_num, how="inner")
        elif query_config['name'] == 'census_self_join':
            join_result = sample_list0.join(sample_list1, sample_list0.a_education_num==sample_list1.b_education_num, how="inner")
        elif query_config['name'] == 'flights_self_join':
            join_result = sample_list0.join(sample_list1, sample_list0.a_unique_carrier==sample_list1.b_unique_carrier, how="inner")
        elif query_config['name'] == 'customer_join_nation':
            join_result = sample_list0.join(sample_list1, sample_list0.c_nationkey==sample_list1.n_nationkey, how="inner")
        elif query_config['name'] == 'customer_join_supplier':
            join_result = sample_list0.join(sample_list1, sample_list0.c_nationkey==sample_list1.s_nationkey, how="inner")
        elif query_config['name'] == 'sales_join_store':
            join_result = sample_list0.join(sample_list1, sample_list0.ss_store_sk==sample_list1.s_store_sk, how="inner")
        elif query_config['name'] == 'ssales_join_wsales':
            join_result = sample_list0.join(sample_list1, sample_list0.ss_promo_sk==sample_list1.ws_promo_sk, how="inner")
        else: 
            join_result = sample_list0.join(sample_list1, sample_list0.join_cols[0]==sample_list1.join_cols[1], how="inner")

        # join_result = pd.merge(sample_list[0], sample_list[1], left_on=join_cols[0], right_on=join_cols[1], how='inner')

        if len(join_cols) > 2:
            join_nums = len(join_cols)
            sample_index = 2
            join_index = 2
            while join_nums != 2:
                join_result = pd.merge(join_result, sample_list[sample_index], left_on=join_cols[join_index], right_on=join_cols[join_index+1], how='inner')
                sample_index += 1
                join_index += 2
                join_nums -= 2

        if len(groupby_cols) > 0:  # with group by clause
            if not outlier:
                for col in avg_cols:
                    aggregations[col] = 'mean'

                for col in sum_cols:
                    aggregations['scale_' + col] = 'sum'

                for col in rate_cols:
                    aggregations[col] = 'mean'

                for train_config in train_config_list:
                    for numeric_column in train_config['numeric_columns']:
                        rate = 1
                        for col in join_result.columns:
                            if col.endswith('_rate'):
                                rate *= join_result[col]
                        join_result = join_result.withColumn('scale_' + numeric_column, join_result[numeric_column] / rate)
                
                agg_result = join_result.groupby(groupby_cols).agg(aggregations)
                for col in agg_result.columns:
                    if col.endswith('_rate)'):
                        agg_result = agg_result.drop(col)
                    elif col.startswith('sum(scale_'):
                        agg_result = agg_result.withColumnRenamed(col, 'sum(' + col[10:])
            else:
                for col in avg_cols:
                    agg_name = "avg({})".format(col)
                    aggregations[agg_name] = (col, 'mean')
                for col in sum_cols:
                    agg_name = "sum({})".format(col)
                    aggregations[agg_name] = (col, 'sum')
                for col in rate_cols:
                    agg_name = col
                    aggregations[agg_name] = (col, 'mean')
                    
                cnt_col = 'cnt'
                aggregations[cnt_col] = (avg_cols[0], 'size')
                agg_result = join_result.groupby(by=groupby_cols + rate_cols).agg(**aggregations)
                # agg_result.to_csv('./agg_result.csv')

                rate_col = 'rate'
                agg_result[rate_col] = 1
                for col in agg_result.columns:
                    if col.endswith('_rate'):
                        agg_result[rate_col] *= agg_result[col]
                        del agg_result[col]

                f_aggregations = {}
                for col in agg_result.columns:
                    if col.startswith('sum') or col == 'cnt':
                        agg_result[col] /= agg_result[rate_col]
                        f_aggregations[col] = (col, 'sum')


                agg_result = agg_result.groupby(by=groupby_cols).agg(**f_aggregations)
                for col in avg_cols:
                    magg_name = "avg({})".format(col)
                    sagg_name = "sum({})".format(col)
                    agg_result[magg_name] = agg_result[sagg_name] / agg_result[cnt_col]
                del agg_result[cnt_col]
        else:  # without group by clause
            for col in avg_cols:
                    agg_name = "avg({})".format(col)
                    aggregations[agg_name] = (col, 'mean')
            for col in sum_cols:
                agg_name = "sum({})".format(col)
                aggregations[agg_name] = (col, 'sum')
            for col in rate_cols:
                agg_name = col
                aggregations[agg_name] = (col, 'mean')
            cnt_col = 'cnt'
            aggregations = {agg_name: aggregations[agg_name] for agg_name in aggregations if agg_name.startswith('sum')}
            aggregations[cnt_col] = (avg_cols[0], 'size')
            agg_result = join_result.groupby(by=rate_cols).agg(**aggregations)
            agg_result.reset_index(inplace=True)
            # print(agg_result)
            rate_col = 'rate'
            agg_result[rate_col] = 1
            for col in agg_result.columns:
                if col.endswith('_rate'):
                    agg_result[rate_col] *= agg_result[col]
                    del agg_result[col]
            # print(outlier)
            # print(agg_result)
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
    else:
        for col in avg_cols:
            agg_name = "avg({})".format(col)
            aggregations[agg_name] = (col, 'mean')
        for col in sum_cols:
            agg_name = "sum({})".format(col)
            aggregations[agg_name] = (col, 'sum')
        for col in rate_cols:
            agg_name = col
            aggregations[agg_name] = (col, 'mean')
        samples = sample_list[0]
        agg_result = samples.groupby(by=groupby_cols).agg(**aggregations)
        rate_col = rate_cols[0]
        for col in agg_result.columns:
            if col.startswith('sum'):
                # if col.endswith('_sum'):
                agg_result[col] /= agg_result[rate_col]
        del agg_result[rate_col]
    # save samples, but bring I/O cost
    # agg_result.to_csv('./samples/agg_results/'+query_config['name']+'.csv',index=False)
    return agg_result


def sample_generation_and_aggregation(model_dataset_list, query_config, train_config_list, sample_agg_list, createDataFrame_time_list):
    start_time = time.perf_counter()
    sample_list = generate_sample_list(model_dataset_list, query_config, train_config_list)
    createDataFrame_time = [0]
    sample_agg = sample_aggregation(sample_list, query_config, train_config_list, createDataFrame_time)
    sample_agg_list.append(sample_agg)
    createDataFrame_time_list.append(createDataFrame_time[0])
    end_time = time.perf_counter()
    logger.info('sample and aggregation time elapsed:{}'.format(end_time - start_time))

def model_aqp(query_config, train_config_list):
    model_dataset_list = train_load_models(train_config_list)
    multi_sample_times = query_config['multi_sample_times']
    groupby_cols = query_config['groupby_cols']
    sample_agg_list = []
    createDataFrame_time_list = []
    threads = []
    # start_time = time.perf_counter()
    for i in range(multi_sample_times):
        logger.info("multi_sampling No.{} epoch".format(i))

        thread = threading.Thread(target=sample_generation_and_aggregation,
                                  args=(model_dataset_list, query_config, train_config_list, sample_agg_list, createDataFrame_time_list))
        threads.append(thread)
        thread.start()

    sample_time = 0
    start_time = time.perf_counter()
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    sample_time = end_time - start_time - createDataFrame_time_list[-1]
    
    for i in range(multi_sample_times):
        sample_agg_list[i] = sample_agg_list[i].toPandas()
        sample_agg_list[i].set_index(query_config['groupby_cols'], inplace=True)

    # if len(groupby_cols) > 0:
    #     sample_agg = pd.concat(sample_agg_list).groupby(groupby_cols).mean()
    # else: 
    #     sample_agg = pd.concat(sample_agg_list).mean()

    start_time = time.perf_counter()
    sample_agg = pd.concat(sample_agg_list).groupby(level=0).mean()
    end_time = time.perf_counter()
    logger.info("sample time: {}".format(sample_time + end_time - start_time))
    
    index_flag = True
    if len(query_config['groupby_cols']) == 0:
        index_flag = False
    diff = compare_aggregation(sample_agg, query_config, index_flag)
    diff_norm = compare_aggregation_norm(sample_agg, query_config, index_flag)

    logger.info("relative error:\n{}".format(diff[:50]))
    logger.info("relative error normalized:\n{}".format(diff_norm[:50]))

    logger.info("relative error average: {}".format(diff.values.sum() / diff.size))
    logger.info("relative error normalized average: {}".format(diff_norm.values.sum() / diff_norm.size))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    # filename='./new_logs/exp1.2/census/tmpp.log',
                    # filename=log_file, 
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    start_time = time.perf_counter()
    query_config_file = sys.argv[1]
    with open(query_config_file) as f:
        query_config = json.load(f)
        logger.info("load query config {} successfully".format(query_config_file))
    train_config_files = query_config['train_config_files']
    train_config_list = []
    for config_file in train_config_files:
        with open(config_file) as f:
            train_config = json.load(f)
            train_config_list.append(train_config)
            logger.info("load train config {} successfully".format(config_file))

    op = query_config['operation']
    if op == 'origin':
        # ground truth
        ground_truth_aggregation(query_config, train_config_list)
    elif op == 'uniform':
        # uniform sample
        uniform_aqp(query_config, train_config_list)
    else:
        ## model aqp
        model_aqp(query_config, train_config_list)
        end_time = time.perf_counter()
        logger.info("total_time:{}".format(end_time - start_time))
