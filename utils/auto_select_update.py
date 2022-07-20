import pandas as pd
import numpy as np
import sys
from statistics import mean
from scipy.stats import ks_2samp

# flights1: python auto_select_update.py flights1 /home/lihan/train_dataset/flights/flight-a.csv /home/lihan/train_dataset/flights/flight-inc.csv a_taxi_out a_air_time a_distance
# tpcds1: python auto_select_update.py tpcds1 /home/lihan/train_dataset/tpcds_0.6667g/store_sales.csv /home/lihan/train_dataset/tpcds_0.6667g/store_sales_inc.csv ss_wholesale_cost ss_list_price
# tpcds2: python auto_select_update.py tpcds2 /home/lihan/train_dataset/tpcds_0.6667g/store_sales.csv /home/lihan/train_dataset/tpcds_0.6667g/store_sales_inc.csv ss_wholesale_cost ss_list_price ss_sales_price ss_ext_sales_price



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