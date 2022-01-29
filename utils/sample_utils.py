import math
import random
import json
import numpy as np
import pandas as pd
from pandasql import sqldf
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
import threading
from multiprocessing import Pool, Process

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def to_2_power(n):
    n = n - 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def nested_group_sample(x, cnt_col):
    sample_cnt = len(x)
    while sample_cnt > 1:
        data = x[x[cnt_col] == sample_cnt]
        sample_cnt = sample_cnt >> 1
        idx = data.sample(sample_cnt).index
        x.loc[idx, cnt_col] = sample_cnt
    return x


def extract_group_sample(x, join_col, cnt_col, sample_cnt):
    group_value = x.iloc[0][join_col]
    power_num = to_2_power(sample_cnt[group_value])
    power_num = power_num if power_num <= len(x) else power_num  # len(x) power_num >> 1
    return x[x[cnt_col] <= power_num]


def extract_sample(samples, join_col, cnt_col, sample_cnt):
    ex_samples = samples.groupby(join_col).apply(lambda x: extract_group_sample(x, join_col, cnt_col, sample_cnt))
    ex_samples.reset_index(inplace=True, drop=True)
    return ex_samples


def nested_sample(df, join_col, cnt_col):
    samples = df.groupby(join_col).apply(lambda x: nested_group_sample(x, cnt_col))
    return samples


def group_sample(x, join_col, cnt_col, sample_cnt):
    group_value = x.iloc[0][join_col]
    power_num = to_2_power(sample_cnt[group_value])
    power_num = power_num if power_num <= len(x) else power_num >> 1  # len(x)  power_num >> 1
    samples = x.sample(power_num)
    samples[cnt_col] = power_num
    return samples


def join_sample(df, join_col, cnt_col, sample_cnt):
    # join_value_count = df[join_col].value_counts().sort_index()
    # join_sample_count = (join_value_count * frac).astype(int)
    # join_sample_count = join_sample_count.apply(to_2_power)
    # total = len(df) * frac
    # k = int(total / len(join_value_count))
    # samples = df.groupby(join_col).apply(lambda x: x.sample(to_2_power(int(len(x) * frac))))
    # samples = df.groupby(join_col).apply(lambda x: x.sample(k if len(x) > k else len(x)))
    samples = df.groupby(join_col).apply(lambda x: group_sample(x, join_col, cnt_col, sample_cnt))
    samples.reset_index(inplace=True, drop=True)
    return samples


def control_group():
    start_time = time.perf_counter()
    path = "../datasets/tpch-1-imba/customer.csv"
    c_cols = ["c_nationkey", "c_acctbal"]
    customer = pd.read_csv(path)[c_cols]
    # print(customer[c_cols[0]].value_counts().sort_index())

    path = "../datasets/tpch-1-imba/supplier.csv"
    s_cols = ["s_nationkey", "s_acctbal"]
    supplier = pd.read_csv(path)[s_cols]

    # customer[c_cols[0]].value_counts().sort_index().plot(kind='bar')
    # plt.show()
    # supplier[s_cols[0]].value_counts().sort_index().plot(kind='bar')
    # plt.show()
    c_frac = 0.1
    s_frac = 0.2
    c_sample = customer.sample(frac=c_frac)
    s_sample = supplier.sample(frac=s_frac)
    cs_sample = pd.merge(c_sample, s_sample, how='inner', left_on='c_nationkey', right_on='s_nationkey')

    vc = pd.read_csv("../datasets/tpch-1-imba/customer_supplier_count.csv", squeeze=True)
    ax = vc.plot(kind='bar')
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.show()
    print(vc)
    vc_sample = cs_sample[c_cols[0]].value_counts().sort_index()
    vc_sample.plot(kind='bar')
    plt.show()
    rate = vc_sample / vc

    print("sample join done")
    print(customer[c_cols[0]].value_counts().sort_index())
    print(supplier[s_cols[0]].value_counts().sort_index())
    print(c_sample[c_cols[0]].value_counts().sort_index())
    print(s_sample[s_cols[0]].value_counts().sort_index())
    print(cs_sample[c_cols[0]].value_counts().sort_index())
    path = "../datasets/tpch-1-imba/customer_supplier_result.csv"
    origin_result = pd.read_csv(path)

    print(origin_result)
    pysqldf = lambda q: sqldf(q, globals())

    sample_result = cs_sample.groupby(by=c_cols[0]).agg(cagg_sum=(c_cols[1], 'sum'), cagg_mean=(c_cols[1], 'mean'),
                                                        sagg_sum=(s_cols[1], 'sum'), sagg_mean=(s_cols[1], 'mean'))
    sample_result['rate'] = c_frac * s_frac
    sample_result['cagg_sum'] = sample_result['cagg_sum'] / sample_result['rate']
    sample_result['sagg_sum'] = sample_result['sagg_sum'] / sample_result['rate']
    print(sample_result)
    del sample_result['rate']
    diff = (origin_result - sample_result).abs() / origin_result
    diff.fillna(1, inplace=True)
    print(diff)
    diff.to_csv("../test/diff.csv")
    print("total error:{}".format(diff.values.sum() / diff.size))
    end_time = time.perf_counter()
    print("time elapsed:{}".format(end_time - start_time))

    # join=pysqldf(sql)
    # join=pd.merge(customer,supplier,how='inner',left_on='c_nationkey',right_on='s_nationkey')
    # join.to_csv("../datasets/tpch-1-imba/customer_supplier.csv",index=False)

    # sd = supplier[supplier[s_cols[0]] == 2].sample(frac=0.9)
    # supplier = supplier.drop(sd.index)
    # sd = supplier[supplier[s_cols[0]] == 3].sample(frac=0.9)
    # supplier = supplier.drop(sd.index)
    # path = "../datasets/tpch-1-imba/supplier.csv"
    # supplier.to_csv(path, index=False)
    # print(supplier[s_cols[0]].value_counts().sort_index())


def compare(sample_result):
    path = "../datasets/tpch-1-imba/customer_supplier_result.csv"
    origin_result = pd.read_csv(path)
    print(origin_result)
    diff = (origin_result - sample_result).abs() / origin_result
    diff.fillna(1, inplace=True)
    # print(diff)
    # diff.to_csv("../test/diff.csv")
    print("total error:{}".format(diff.values.sum() / diff.size))
    return diff


def cal(ex_c, ex_s, vc, sample_result_list):
    start_time = time.perf_counter()
    cal_start = time.perf_counter()
    ex_cc = ex_c.groupby(c_cols[0])[cnt_col].max()
    ex_sc = ex_s.groupby(s_cols[0])[cnt_col].max()
    svc = ex_cc * ex_sc
    # cnt = c_sample.groupby([c_cols[0], 'sample_cnt']).count()
    # print(cnt)
    cs_sample = pd.merge(ex_c, ex_s, how='inner', left_on='c_nationkey', right_on='s_nationkey')
    # svc = cs_sample[c_cols[0]].value_counts().sort_index()
    # vc.plot(kind='bar')
    # plt.show()
    # svc.plot(kind='bar')
    # plt.show()

    rate = svc / vc
    # print(origin_result)
    sample_result = cs_sample.groupby(by=c_cols[0]).agg(cagg_sum=(c_cols[1], 'sum'), cagg_mean=(c_cols[1], 'mean'),
                                                        sagg_sum=(s_cols[1], 'sum'), sagg_mean=(s_cols[1], 'mean'))
    sample_result['rate'] = rate
    sample_result['cagg_sum'] = sample_result['cagg_sum'] / sample_result['rate']
    sample_result['sagg_sum'] = sample_result['sagg_sum'] / sample_result['rate']
    cal_end = time.perf_counter()
    print("cal time elapsed:{}".format(cal_end - cal_start))

    # print(sample_result)
    del sample_result['rate']
    sample_result_list.append(sample_result)
    diff = compare(sample_result)
    # d0.append(diff.loc[[0]])
    # d1.append(diff.loc[[1]])
    # d2.append(diff.loc[[2]])
    # d3.append(diff.loc[[3]])
    # d0['cagg_sum']=diff.loc[0,'cagg_sum']
    # d0['cagg_mean'] = diff.loc[0, 'cagg_mean']
    # d0['sagg_sum'] = diff.loc[0, 'sagg_sum']
    # d0['sagg_mean'] = diff.loc[0, 'sagg_mean']
    # d1['cagg_sum'] = diff.loc[1, 'cagg_sum']
    # d1['cagg_mean'] = diff.loc[1, 'cagg_mean']
    # d1['sagg_sum'] = diff.loc[1, 'sagg_sum']
    # d1['sagg_mean'] = diff.loc[1, 'sagg_mean']
    end_time = time.perf_counter()
    print("one sample round time elapsed:{}".format(end_time - start_time))


def sample(customer, supplier, c_cols, s_cols, cnt_col, c_cnt, s_cnt, c_samples, s_samples):
    sample_start = time.perf_counter()
    c_sample = join_sample(customer, c_cols[0], cnt_col, c_cnt)
    s_sample = join_sample(supplier, s_cols[0], cnt_col, s_cnt)
    # cjc = c_sample.groupby([c_cols[0], cnt_col]).size()
    # sjc = s_sample.groupby([s_cols[0], cnt_col]).size()
    c_sample = nested_sample(c_sample, c_cols[0], cnt_col)
    s_sample = nested_sample(s_sample, s_cols[0], cnt_col)
    c_cnt = c_sample[c_cols[0]].value_counts().sort_index().to_dict()
    # c_cnt[2] = 128
    # c_cnt[3] = 128
    s_cnt = s_sample[s_cols[0]].value_counts().sort_index().to_dict()
    # s_cnt[0] = 32
    # s_cnt[1] = 32
    ex_c = extract_sample(c_sample, c_cols[0], cnt_col, c_cnt)
    ex_s = extract_sample(s_sample, s_cols[0], cnt_col, s_cnt)
    c_samples.append(ex_c)
    s_samples.append(ex_s)
    sample_end = time.perf_counter()
    print("sample time elapsed:{}".format(sample_end - sample_start))


def origin_query(customer, supplier):
    group_col = 'c_nationkey'
    cagg_col = 'c_acctbal'
    sagg_col = 's_acctbal'
    cs = pd.merge(customer, supplier, how='inner', left_on='c_nationkey', right_on='s_nationkey')
    cs_res = cs.groupby(by=group_col).agg(cagg_sum=(cagg_col, 'sum'), cagg_mean=(cagg_col, 'mean'),
                                          sagg_sum=(sagg_col, 'sum'), sagg_mean=(sagg_col, 'mean'))
    print(cs_res)


def stratified_sample(df, groupby_col, sample_allocation):
    samples = df.groupby(groupby_col).apply(lambda x: x.sample(sample_allocation[x.name]))
    return samples

def aggregation(df,groupby_col,sum_cols,avg_cols):
    aggs={}
    for col in sum_cols:
        agg_name = col + "_sum"
        aggs[agg_name] = (col, 'sum')

    for col in avg_cols:
        agg_name = col + "_mean"
        aggs[agg_name] = (col, 'mean')
    res=df.groupby(groupby_col).agg(**aggs)
    return res

def test(x):
    return x.name


if __name__ == '__main__':
    start_time = time.perf_counter()
    # query_config_file = './config/query/customer_join_supplier.json'
    query_config_file = './config/query/customer.json'
    with open(query_config_file) as f:
        query_config = json.load(f)
        print("load query config {} successfully".format(query_config_file))
    train_flag = query_config['train_flag']

def multiple_sample_test():
    # control_group()
    total_start_time = time.perf_counter()
    path = "../datasets/tpch-1/customer.csv"
    c_cols = ["c_nationkey", "c_acctbal", "c_mktsegment"]
    customer = pd.read_csv(path)[c_cols]
    customer.groupby(c_cols[0]).apply(lambda x: test(x))
    # print(customer[c_cols[0]].value_counts().sort_index())

    path = "../datasets/tpch-1/supplier.csv"
    s_cols = ["s_nationkey", "s_acctbal"]
    supplier = pd.read_csv(path)[s_cols]
    # origin_query(customer,supplier)
    # total_end_time = time.perf_counter()
    # print("total time elapsed:{}".format(total_end_time - total_start_time))

    cc = customer[c_cols[0]].value_counts().sort_index()
    # ax1 = cc.plot(kind='bar')
    # plt.show()
    sc = supplier[s_cols[0]].value_counts().sort_index()
    # ax2 = sc.plot(kind='bar')
    # plt.show()
    c_frac = 0.1
    s_frac = 0.2
    # vvc = pd.read_csv("../datasets/tpch-1-imba/customer_supplier_count.csv", squeeze=True)
    vc = cc * sc
    # print(sum(vc))
    # c_total = sum(cc)
    # c_groups = len(cc)
    # c_threshold = 1 / c_groups / 5 * c_total
    # c_cnt = (cc).apply(lambda x: math.ceil(c_frac * x) if x > c_threshold else x).to_dict()
    c_cnt = (cc).apply(lambda x: math.ceil(c_frac * x)).to_dict()

    # s_total = sum(sc)
    # s_groups = len(sc)
    # s_threshold = (1 / s_groups) / 5 * s_total
    # s_cnt = (sc).apply(lambda x: math.ceil(s_frac * x) if x > s_threshold else x).to_dict()
    s_cnt = (sc).apply(lambda x: math.ceil(s_frac * x)).to_dict()
    cnt_col = 'sample_cnt'
    for i in range(1):
        sample_times = 1
        sample_result_list = []
        # d0 = {'cagg_sum': [], 'cagg_mean': [], 'sagg_sum': [], 'sagg_mean': []}
        # d1 = {'cagg_sum': [], 'cagg_mean': [], 'sagg_sum': [], 'sagg_mean': []}
        d0 = []
        d1 = []
        d2 = []
        d3 = []
        threads = []
        c_samples = []
        s_samples = []
        # for _ in range(sample_times):
        # thread = threading.Thread(target=sample,
        #                           args=(
        #                           customer, supplier, c_cols, s_cols, cnt_col, c_cnt, s_cnt, vc, sample_result_list))
        # threads.append(thread)
        # thread.start()
        # sample(customer, supplier, c_cols, s_cols, cnt_col, c_cnt, s_cnt, c_samples, s_samples)

        group_col = 'c_nationkey'
        cagg_col = 'c_acctbal'
        sagg_col = 's_acctbal'
        sc = customer.groupby(group_col).apply(lambda x: x.sample(int(c_frac * len(x))))
        sc.reset_index(inplace=True, drop=True)
        ss = supplier.groupby('s_nationkey').apply(lambda x: x.sample(int(s_frac * len(x))))
        ss.reset_index(inplace=True, drop=True)
        print(sc.groupby(group_col).size())
        print(ss.groupby('s_nationkey').size())
        cs = pd.merge(sc, ss, how='inner', left_on='c_nationkey', right_on='s_nationkey')
        print(cs.groupby(group_col).size())
        print(vc)
        cs_res = cs.groupby(by=group_col).agg(cagg_sum=(cagg_col, 'sum'), cagg_mean=(cagg_col, 'mean'),
                                              sagg_sum=(sagg_col, 'sum'), sagg_mean=(sagg_col, 'mean'))
        print(cs_res)
        cs_res['rate'] = c_frac * s_frac
        cs_res['cagg_sum'] = cs_res['cagg_sum'] / cs_res['rate']
        cs_res['sagg_sum'] = cs_res['sagg_sum'] / cs_res['rate']
        del cs_res['rate']

        ocs = pd.merge(customer, supplier, how='inner', left_on='c_nationkey', right_on='s_nationkey')
        ocs_res = ocs.groupby(by=group_col).agg(cagg_sum=(cagg_col, 'sum'), cagg_mean=(cagg_col, 'mean'),
                                                sagg_sum=(sagg_col, 'sum'), sagg_mean=(sagg_col, 'mean'))
        print(cs_res)
        print(ocs_res)
        diff = (ocs_res - cs_res).abs() / ocs_res
        diff.fillna(1, inplace=True)
        print(diff)
        print("total error:{}".format(diff.values.sum() / diff.size))

        # cs_res.to_csv("./output/cs_result.csv", index=False)
        #     for t in threads:
        #         t.join()
        #     # for ex_c in c_samples:
        #     #     for ex_s in s_samples:
        #     #         cal(ex_c, ex_s, vc, sample_result_list)
        #
        #     for i in range(len(c_samples)):
        #         cal(c_samples[i], s_samples[i], vc, sample_result_list)
        #
        #     # pd.concat(d0).to_csv('../test/d0.csv', index=False)
        #     # pd.concat(d1).to_csv('../test/d1.csv', index=False)
        #     # pd.concat(d2).to_csv('../test/d2.csv', index=False)
        #     # pd.concat(d3).to_csv('../test/d3.csv', index=False)
        #
        #     agg_sample_result = sum(sample_result_list)
        #     agg_sample_result = agg_sample_result / len(sample_result_list)
        #     diff = compare(agg_sample_result)
        #     print(diff)
        #     diff.loc[[0]].to_csv("../test/dff0.csv", mode='a', header=False, index=False)
        #     diff.loc[[1]].to_csv("../test/dff1.csv", mode='a', header=False, index=False)
        #     diff.loc[[2]].to_csv("../test/dff2.csv", mode='a', header=False, index=False)
        #     diff.loc[[3]].to_csv("../test/dff3.csv", mode='a', header=False, index=False)
        total_end_time = time.perf_counter()
        print("total time elapsed:{}".format(total_end_time - total_start_time))
