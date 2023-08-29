import pandas as pd
import numpy as np
import sys
from statistics import mean

# flights1: python auto_select_encoder.py flights1 /root/lihan/train_dataset/flights/flight-a.csv a_taxi_out a_air_time a_distance
# census1: python auto_select_encoder.py census1 /root/lihan/train_dataset/census/adult-a.csv a_fnlwgt a_age a_hours_per_week
# census2: python auto_select_encoder.py census2 /root/lihan/train_dataset/census/adult-a.csv a_fnlwgt a_age a_hours_per_week
# tpch1: python auto_select_encoder.py tpch1_customer /home/lihan/train_dataset/tpch_20g/customer.csv c_acctbal
#        python auto_select_encoder.py tpch1_supplier /root/lihan/train_dataset/tpch_35g/supplier.csv s_acctbal
# tpcds1: python auto_select_encoder.py tpcds1_ssales /root/lihan/train_dataset/tpcds_0.6667g/store_sales.csv ss_wholesale_cost ss_list_price
# tpcds1: python auto_select_encoder.py tpcds1_wsales /root/lihan/train_dataset/tpcds_0.6667g/web_sales.csv ws_wholesale_cost ws_list_price
# tpcds2: python auto_select_encoder.py tpcds2 /root/lihan/train_dataset/tpcds_0.6667g/store_sales.csv ss_wholesale_cost ss_list_price ss_sales_price ss_ext_sales_price



dataset = sys.argv[1]           
file_path = sys.argv[2]         
numeric_columns = sys.argv[3:]  

data = pd.read_csv(file_path, delimiter=',')
data_simple = data[numeric_columns]
# print(data_simple.describe())

coefficient = []

for numeric_column in numeric_columns:
    segments=pd.cut(data_simple[numeric_column],bins=9)
    counts=pd.value_counts(segments,sort=False)
    coefficient.append(counts.std()/counts.mean())

print(coefficient)
print(mean(coefficient))

if mean(coefficient) < 1.3845437937603253:
    print("Selected encoded method: mm")
else:
    print("Selected encoded method: gaussian")