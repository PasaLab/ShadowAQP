import os
import sys
import argparse
import re
import json

sql_path = './query/'

parser = argparse.ArgumentParser(description='command line parsing')
parser.add_argument('sql_file', type=str, help='input sql file')
parser.add_argument('flag', type=str, help='train/load flag')
parser.add_argument('dataset_path', type=str, help='input dataset path, separate with commas')
parser.add_argument('ground_truth', type=str, help='ground truth path')
parser.add_argument('cat_attr1', type=str, help='categorical attributes of table1, separate with commas between attributes')
parser.add_argument('cat_attr2', type=str, help='categorical attributes of table2, separate with commas between attributes')
parser.add_argument('num_attr1', type=str, help='number attributes of table1, separate with commas between attributes')
parser.add_argument('num_attr2', type=str, help='number attributes of table2, separate with commas between attributes')
parser.add_argument('sampling_ratio', type=str, help='sampling ratios of tables, separate with commas')
parser.add_argument('delimiter1', type=str, help='delimiter in the dataset of table1')
parser.add_argument('delimiter2', type=str, help='delimiter in the dataset of table2')


args = parser.parse_args()
with open(sql_path + args.sql_file) as f:
    sqltxt = f.readlines()

sql = "".join(sqltxt[1]).split()
table1, table2 = sql[1], sql[3]

sql = "".join(sqltxt[2]).split()
join_attr1, join_attr2 = sql[1], sql[3]

sql = "".join(sqltxt[3]).split()
groupby_attr = sql[2]

p = re.compile(r'[(](.*?)[)]')
sql = "".join(sqltxt[0]).split()
agg_col = []
for i in range(2, len(sql)):
    result = re.findall(p, sql[i])
    if result[0] not in agg_col:
        agg_col.append(result[0])

label_columns = [join_attr1]
if join_attr1 != groupby_attr and join_attr1[0] == groupby_attr[0]:
    label_columns = [join_attr1, groupby_attr]

query_name = table1 + '_join_' + table2
train_name1 = query_name + '-' + table1
train_name2 = query_name + '-' + table2
query_conf_file = 'generate_config/query/' + query_name + '.json'
train_conf_file1 = 'generate_config/train/' + train_name1 + '.json'
train_conf_file2 = 'generate_config/train/' + train_name2 + '.json'

query_conf = {"name": query_name,
              "train_config_files": [train_conf_file1, train_conf_file2],
              "multi_sample_times": 3,
              "operation": "aqp",
              "join_cols": [join_attr1, join_attr2],
              "groupby_cols": [groupby_attr],
              "sum_cols": agg_col,
              "avg_cols": agg_col,
              "ground_truth": args.ground_truth}

train_conf1 = {"name": train_name1,
               "data": args.dataset_path.split(',')[0],
               "categorical_columns": args.cat_attr1,
               "numeric_columns": args.num_attr1,
               "label_columns": [join_attr1],
               "bucket_columns": label_columns,
               "categorical_encoding": "binary",
               "numeric_encoding": "gaussian",
               "max_clusters": 5,
               "model_type": "torch_cvae",
               "lr": 0.001,
               "optimizer_type": "adam",
               "loss_agg_type": "mean",
               "gpu_num": 0,
               "epochs": 150,
               "batch_size": 1024,
               "latent_dim": 50,
               "intermediate_dim": 100,
               "train_flag": args.flag,
               "operation": "aqp",
               "sample_method": "statistics",
               "sample_rate": args.sampling_ratio.split(',')[0],
               "sample_for_train": 1,
               "header": 1,
               "delimiter": args.delimiter1}

train_conf2 = {"name": train_name2,
               "data": args.dataset_path.split(',')[1],
               "categorical_columns": args.cat_attr2,
               "numeric_columns": args.num_attr2,
               "label_columns": [join_attr2],
               "bucket_columns": [],
               "categorical_encoding": "binary",
               "numeric_encoding": "mm",
               "max_clusters": 5,
               "model_type": "torch_cvae",
               "lr": 0.001,
               "optimizer_type": "adam",
               "loss_agg_type": "mean",
               "gpu_num": 0,
               "epochs": 100,
               "batch_size": 128,
               "latent_dim": 100,
               "intermediate_dim": 150,
               "train_flag": args.flag,
               "operation": ["aqp" if args.sampling_ratio.split(',')[1] != 1 else "other"],
               "sample_method": "house",
               "sample_rate": args.sampling_ratio.split(',')[1],
               "sample_for_train": 1,
               "header": 1,
               "delimiter": args.delimiter2}


json.dump(query_conf, open(query_conf_file,'w'),indent=4)
json.dump(train_conf1, open(train_conf_file1,'w'),indent=4)
json.dump(train_conf2, open(train_conf_file2,'w'),indent=4)