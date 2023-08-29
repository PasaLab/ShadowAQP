import pandas as pd
import numpy as np
import sys
from statistics import mean
from scipy.stats import ks_2samp

# flights1: python auto_select_update.py flights1 /home/lihan/train_dataset/flights/flight-a.csv /home/lihan/train_dataset/flights/flight-inc.csv a_taxi_out a_air_time a_distance
# tpcds1: python auto_select_update.py tpcds1 /home/lihan/train_dataset/tpcds_0.6667g/store_sales.csv /home/lihan/train_dataset/tpcds_0.6667g/store_sales_inc.csv ss_wholesale_cost ss_list_price
# tpcds2: python auto_select_update.py tpcds2 /home/lihan/train_dataset/tpcds_0.6667g/store_sales.csv /home/lihan/train_dataset/tpcds_0.6667g/store_sales_inc.csv ss_wholesale_cost ss_list_price ss_sales_price ss_ext_sales_price
# tpch_cn: python auto_select_update.py tpch_cn /home/lihan/train_dataset/tpch_20g/customer.csv /home/lihan/train_dataset/tpch_20g/customer_inc.csv c_acctbal
# census1: python auto_select_update.py census1 /home/lihan/train_dataset/census/adult-a.csv /home/lihan/train_dataset/census/adult-inc.csv a_fnlwgt a_age a_hours_per_week
# census2: python auto_select_update.py census2 /home/lihan/train_dataset/census/adult-a.csv /home/lihan/train_dataset/census/adult-inc.csv a_fnlwgt a_age a_hours_per_week

# tpcds1:   6.714966769569961e-07       sample_train
# tpcds2:   0.0496741643213764          inc_train
# flights1: 0.871549633956838           inc_train
# tpch_cn:  0.43968781269736174         inc_train
# census1:  0.7337213739305167          inc_train
# census2:  0.7337213739305167          inc_train



dataset = sys.argv[1]           
origin_file_path = sys.argv[2]
inc_file_path = sys.argv[3]
agg_cols = sys.argv[4:]  

origin_data = pd.read_csv(origin_file_path, delimiter=',')
inc_data = pd.read_csv(inc_file_path, delimiter=',')

p_values = []

for agg_col in agg_cols:
    p_values.append(ks_2samp(origin_data[agg_col], inc_data[agg_col])[1])

print(dataset, ': ', p_values)

print('mean(p_values): ', mean(p_values))

if mean(p_values) < 0.01:
    print("Selected update strategy: sample_train")
else:
    print("Selected update strategy: inc_train")