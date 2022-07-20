import pandas as pd
import numpy as np
from pandasql import sqldf
import time
import category_encoders as ce
np.set_printoptions(suppress=True)

def execute_sql(path, sql):
    df = pd.read_csv(path)
    result = sqldf(sql, locals())
    return result


def execute_multitables_sql(paths, sql):
    start_time = time.perf_counter()
    names = locals()
    for path in paths:
        name = path.split("/")[-1]
        name = name.split('.')[0]
        names[name] = pd.read_csv(path)
    result = sqldf(sql, locals())
    end_time = time.perf_counter()
    print('[INFO]execute query time:{}'.format(end_time - start_time))
    return result


def scale_result(result, fraction):
    return result / fraction


def compare_result(origin_result, aqp_result):
    (rows, cols) = origin_result.shape
    count = rows * cols
    compare = ((origin_result - aqp_result).abs()) / origin_result
    print(compare)
    compare.fillna(1, inplace=True)
    result = compare.values.sum() / count
    return result


if __name__ == '__main__':
    ### compare customer
    # origin_path = '../datasets/tpch-0.1/customer.csv'
    # samples_path = '../output/keras_cvae_customer_c_nationkey_ld32_id64_bs16_ep100.csv'
    # sql = 'select sum(c_acctbal) from df group by c_nationkey'

    origin_path = '../datasets/sdr/sdr_flow_ter_15min0001.csv'
    dim_path = '../datasets/sdr/dim_sub_prot.csv'
    samples_path = '../output/keras_cvae_sdr_flow_prot_category_prot_type_ld32_id64_bs16_ep100.csv'
    # origin_sql = 'select PROT_CATEGORY,sum(l4_ul_packets) from sdr_flow_ter_15min0001 join dim_sub_prot on PROT_CATEGORY=PROTOCOL_ID group by PROT_CATEGORY'
    # aqp_sql='select PROT_CATEGORY,sum(l4_ul_packets) from keras_cvae_sdr_flow_prot_category_prot_type_ld32_id64_bs16_ep100 join dim_sub_prot on PROT_CATEGORY=PROTOCOL_ID group by PROT_CATEGORY'
    origin_sql = 'select PROT_CATEGORY,sum(l4_ul_throughput) from sdr_flow_ter_15min0001 group by PROT_CATEGORY'
    aqp_sql='select PROT_CATEGORY,sum(l4_ul_throughput) from keras_cvae_sdr_flow_prot_category_prot_type_ld32_id64_bs16_ep100 group by PROT_CATEGORY'

    origin_paths = [origin_path, dim_path]
    origin_result = execute_multitables_sql(origin_paths, origin_sql)
    aqp_paths = [samples_path, dim_path]
    aqp_result = execute_multitables_sql(aqp_paths, aqp_sql)
    aqp_result = scale_result(aqp_result, fraction=0.01)
    print(origin_result)
    print(aqp_result)
    compare_result = compare_result(origin_result, aqp_result)
    print(compare_result)

    ### compare supplier
    # origin_path = '../datasets/tpch-0.1/supplier.csv'
    # samples_path = '../output/keras_cvae_supplier_z_acctbal_ld32_id64_bs16_ep100.csv'
    # sql = 'select sum(s_acctbal) from df group by s_nationkey'
    # origin_result = execute_sql(origin_path, sql)
    # aqp_result = execute_sql(samples_path, sql)
    # aqp_result = scale_result(aqp_result, fraction=0.1)

    # compare_result = compare_result(origin_result, aqp_result)
    # print(origin_result)
    # print(aqp_result)
    # print(compare_result)
