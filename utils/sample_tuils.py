import math
import numpy as np
import pandas as pd
from pandasql import sqldf
import matplotlib as mlp
import matplotlib.pyplot as plt
import time

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def to_2_power(n):
    n = n - 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def group_sample(x, join_col, sample_cnt):
    group_value = x.iloc[0][join_col]
    power_num = to_2_power(sample_cnt[group_value])
    power_num = power_num if power_num <= len(x) else power_num>>1  # len(x)  power_num >> 1
    samples = x.sample(power_num)
    samples['sample_cnt'] = power_num
    return samples


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
    power_num = power_num if power_num <= len(x) else power_num # len(x) power_num >> 1
    return x[x[cnt_col] <= power_num]


def extract_sample(samples, join_col, cnt_col, sample_cnt):
    ex_samples = samples.groupby(join_col).apply(lambda x: extract_group_sample(x, join_col, cnt_col, sample_cnt))
    ex_samples.reset_index(inplace=True, drop=True)
    return ex_samples


def nested_sample(df, join_col, cnt_col):
    samples = df.groupby(join_col).apply(lambda x: nested_group_sample(x, cnt_col))
    return samples


def join_sample(df, join_col, sample_cnt):
    # join_value_count = df[join_col].value_counts().sort_index()
    # join_sample_count = (join_value_count * frac).astype(int)
    # join_sample_count = join_sample_count.apply(to_2_power)
    # total = len(df) * frac
    # k = int(total / len(join_value_count))

    # samples = df.groupby(join_col).apply(lambda x: x.sample(to_2_power(int(len(x) * frac))))
    # samples = df.groupby(join_col).apply(lambda x: x.sample(k if len(x) > k else len(x)))
    samples = df.groupby(join_col).apply(lambda x: group_sample(x, join_col, sample_cnt))

    samples.reset_index(inplace=True, drop=True)
    return samples


def control_group():
    start_time = time.clock()
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
    end_time = time.clock()
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


if __name__ == '__main__':
    # control_group()
    start_time = time.clock()
    path = "../datasets/tpch-1-imba/customer.csv"
    c_cols = ["c_nationkey", "c_acctbal"]
    customer = pd.read_csv(path)[c_cols]
    # print(customer[c_cols[0]].value_counts().sort_index())

    path = "../datasets/tpch-1-imba/supplier.csv"
    s_cols = ["s_nationkey", "s_acctbal"]
    supplier = pd.read_csv(path)[s_cols]

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
    c_total = sum(cc)
    c_groups = len(cc)
    c_threshold = 1 / c_groups / 5 * c_total
    # c_cnt = (cc).apply(lambda x: math.ceil(c_frac * x) if x > c_threshold else x).to_dict()
    c_cnt = (cc).apply(lambda x: math.ceil(c_frac * x)).to_dict()

    s_total = sum(sc)
    s_groups = len(sc)
    s_threshold = (1 / s_groups) / 5 * s_total
    # s_cnt = (sc).apply(lambda x: math.ceil(s_frac * x) if x > s_threshold else x).to_dict()
    s_cnt = (sc).apply(lambda x: math.ceil(s_frac * x)).to_dict()

    c_sample = join_sample(customer, c_cols[0], c_cnt)
    s_sample = join_sample(supplier, s_cols[0], s_cnt)
    cnt_col = 'sample_cnt'
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
    path = "../datasets/tpch-1-imba/customer_supplier_result.csv"
    origin_result = pd.read_csv(path)
    # print(origin_result)
    sample_result = cs_sample.groupby(by=c_cols[0]).agg(cagg_sum=(c_cols[1], 'sum'), cagg_mean=(c_cols[1], 'mean'),
                                                        sagg_sum=(s_cols[1], 'sum'), sagg_mean=(s_cols[1], 'mean'))
    sample_result['rate'] = rate
    sample_result['cagg_sum'] = sample_result['cagg_sum'] / sample_result['rate']
    sample_result['sagg_sum'] = sample_result['sagg_sum'] / sample_result['rate']
    print(sample_result)
    del sample_result['rate']
    diff = (origin_result - sample_result).abs() / origin_result
    diff.fillna(1, inplace=True)
    print(diff)
    diff.to_csv("../test/diff.csv")
    print("total error:{}".format(diff.values.sum() / diff.size))
    end_time = time.clock()
    print("time elapsed:{}".format(end_time - start_time))
