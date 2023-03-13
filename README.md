# ShadowAQP

 This is the code repository for the approximate query processing paper titled *'ShadowAQP: Efficient Approximate Group-by and Join Query via Attribute-oriented Sample Size Allocation and Data Generation'*.

 This project is built on an open source machine learning framework [PyTorch](https://pytorch.org/). 

 ## Prerequisites

 * OS CentOS 7.5.1804
 * Apache Spark 2.3.2
 * Hive 3.1.2
 * PyTorch 1.8.0
 * Numpy 1.19.5
 * Pandas 1.1.5
 * Keras 2.6.0
 * Sklearn

 ## Quick Start

 1. Clone source code

     ```
     git clone https://github.com/PasaLab/ShadowAQP.git
     ```

 2. Prepare the raw exact query result (under `/ground_truth`) and data

 3. Config the **query configuration files** `config/query/xxx.json` to define a query and  config the **training configuration files** of involved tables `config/train/xxx.json` to guide the training as in [below](#configuration). 

 4. Train the models

     ```
     python main.py ./config/query/xxx.json  # set the 'train_flag' in config/train/xxx.json to 'train'
     ```

 5. Execute the query
     ```
     python main.py ./config/query/xxx.json  # set the 'train_flag' in config/train/xxx.json to 'load'
     ```


 ## Usage
 ShadowAQP proceeds into two phases: **the model training phase (offline)** and **the sample generation phase (online)**.
   - **The model training phase (offline)**
     * In model traing phase, ShadowAQP learns the underlying probability distribution of a table.
     * The model training phase includes three stages: the labeling stage, the encoding stage, and the learning stage.
   - **The sample generation phase (online)**
     * In sample generation phase, ShadowAQP generates sample tuples based on the learned conditional probability distribution.
     * The sample generation phase includes the sampling stage, the decoding stage, and the execution stage.


 ### Model Training Phase (offline)
 1. Labeling Stage

    - The goal of this stage is to label tuples in the table, which is necessary because traditional conditional generative models can only train on labeled data and learn probability distribution conditioned to the labels.

    - We use the values of the **label attributes** to label the tuples and the label attributes is set in `config/train/xxx.json`.

      ```
      {
        ...
        "categorical_columns": [      // the involved categorical attributes
          "protocol_type",
        ],
        "numeric_columns": [      // the involved numeric attributes
          "upload_throughput",
          "download_throughput"
        ],
        "label_columns": [      // the label attributes
          "protocol_type"
        ]
        ...
      }
      ```

 2. Encoding Stage

    - The target of the encoding stage is to encode tuples together with their labels into a data representation suitable for training conditional generative models. 

    - ShadowAQP will encode the tuples according to the **encoding method** configured in `config/train/xxx.json`.

      ```
      {
        ...
        "categorical_encoding": "binary",     // 'binary' for binary encoding, 'onehot' for one-hot encoding
        "numeric_encoding": "gaussian",       // 'gaussian' for guassian mixture encoding, 'minmax' for minmax normalization
        "max_clusters": 15,               // the number of max gaussian clusters
        ...
      }
      ```

 3. Learning Stage

    -  In the learning stage, the encoded data and labels are fed to the neural network model for training.

    -  ShadowAQP strats training with the **learning parameters** configured in `config/train/xxx.json`.

      ```
      {
        ...
        "lr": 0.001,                // learning rate
        "optimizer_type": "adam",   // optimizer
        "loss_agg_type": "mean",    // aggregate function of loss
        "gpu_num": 1,               // No. of gpu 
        "epochs": 200,              // training epochs
        "batch_size": 1024,         // batch size
        "latent_dim": 200,          // latent dimension of CVAE (hidden layer size)
        "intermediate_dim": 200,    // intermediate dimension of CVAE (hidden layer size)
        ...
      }
      ```

 ### Sample Generation Phase (online)

 1. Sampling Stage

    - In the sampling stage, ShadowAQP generates sample vectors with the latent variables sampled from the latent space and the given labels.

    - The sampling rate and sampling method is also configured in `config/train/xxx.json`.

      ```
      {
        ...
        "sample_rate": 0.05, 
        "sample_method": "statistics",
        ...
      }
      ```

 2. Decoding Stage

    - The decoding stage is responsible for converting the sample data generated from the ShadowAQP model into table tuples. 

    - Decoding is the reverse process of the configured encoding methods in `config/train/xxx.json`.

      ```
      {
        ...
        "categorical_encoding": "binary",     // 'binary' for binary encoding, 'onehot' for one-hot encoding
        "numeric_encoding": "gaussian",       // 'gaussian' for guassian mixture encoding, 'minmax' for minmax normalization
        "max_clusters": 15,               // the number of max gaussian clusters
        ...
      }
      ```

 3. Executing Stage

    - In the execution stage, ShadowAQP executes the queries on the generated samples to obtain the approximate query answers.

 ## Configuration

 There are two types of configuration files: query configuration files and training configuration files.

 Query configuration files is under `/config/query`.An example is given below.

 ```shell
 {
   "name": "ssales_join_wsales",
   "train_config_files": [
     "./config/train/tpcds_ssales_torch_cvae.json",
     "./config/train/tpcds_wsales_torch_cvae.json"
   ],
   "multi_sample_times": 1,
   "operation": "aqp",
   "join_cols": ["ss_promo_sk","ws_promo_sk"],
   "groupby_cols": ["ss_promo_sk"],
   "result_path": "./output/aqp_result/ss_res.csv",
   "diff_path": "./output/diff/ss_diff.csv",
   "sum_cols": ["ss_wholesale_cost","ss_list_price","ws_wholesale_cost","ws_list_price"],
   "avg_cols": ["ss_wholesale_cost","ss_list_price","ws_wholesale_cost","ws_list_price"],
   "var": "./var/tpcds-1m/sw_var.csv",
   "ground_truth": "./ground_truth/tpcds-0.6667g/sw_truth.csv" // Specifies the raw exact query result
 }

 ```

 Training configuration files  is under `/config/train`.An example is given below.

 ```shell
 {
     "name": "tpcds-06667g-ssales",
     "data": "/root/lihan/train_dataset/tpcds_0.6667g/store_sales.csv",
     "categorical_columns": [
         "ss_promo_sk"
     ],
     "numeric_columns": [
         "ss_wholesale_cost",
         "ss_list_price"
     ],
     "label_columns": [
         "ss_promo_sk"
     ],
     "bucket_columns": [],
     "categorical_encoding": "binary",
     "numeric_encoding": "mm",
     "max_clusters": 5,
     "model_type": "torch_cvae",
     "lr": 0.001,
     "optimizer_type": "adam",
     "loss_agg_type": "mean",
     "gpu_num": 0,
     "epochs": 150,
     "batch_size": 512,
     "latent_dim": 100,
     "intermediate_dim": 100,
     "train_flag": "load",
     "operation": "aqp",
     "sample_method": "statistics",
     "sample_rate": 0.01,
     "sample_for_train": 1,
     "header": 1,
     "delimiter": ","
 }
 ```

We also provide a script `conf_script.py` to help generate the configuration files.

```
usage: conf_script.py [-h]
               sql_file flag dataset_path ground_truth cat_attr1 cat_attr2
               num_attr1 num_attr2 sampling_ratio delimiter1 delimiter2

command line parsing

positional arguments:
  sql_file        input sql file
  flag            train/load flag
  dataset_path    input dataset path, separate with commas
  ground_truth    ground truth path
  cat_attr1       categorical attributes of table1, separate with commas
                  between attributes
  cat_attr2       categorical attributes of table2, separate with commas
                  between attributes
  num_attr1       number attributes of table1, separate with commas between
                  attributes
  num_attr2       number attributes of table2, separate with commas between
                  attributes
  sampling_ratio  sampling ratios of tables, separate with commas
  delimiter1      delimiter in the dataset of table1
  delimiter2      delimiter in the dataset of table2

optional arguments:
  -h, --help      show this help message and exit
```

It can be easily used like `python conf_script.py tpch_cn.sql train /xxx/train_dataset/tpch_20g/customer.csv,/xxx/train_dataset/tpch_20g/nation.csv ./ground_truth/tpch-20g/cn_truth.csv c_nationkey n_nationkey,n_name c_acctbal '' 0.01,1 ',' '|'`. Then the configuration files will be automatically generated under `/generate_config/query/`  and `generate_config/train/`.

## Example

 ### TPC-DS example

 * Download the tpcds-kit `git clone https://github.com/gregrahn/tpcds-kit.git`, and generate the data with the 1GB scale factor `./dsdgen -scale 1 -dir ../data/ -force`

 * Generate the ground truth with SparkSQL

 * Configure the query configuration file `config/query/ssales_join_wsales.json`,  training configuration files `/config/train/tpcds_ssales_torch_cvae.json` and `/config/train/tpcds_wsales_torch_cvae.json`

 * Set the 'train_flag' in the training configuration files to 'train' and train the models `python main.py config/query/ssales_join_wsales.json `

 * Then set different sampling ratios in the training configuration files and get result from model (set the 'train_flag' in the training configuration files to 'query') `python main.py config/query/ssales_join_wsales.json `

     ```
     ...
     sample time: 1.5985828302800655
     relative error average: 0.0519204122033996
     relative error normalized average: 0.04979657492564337
     total_time:4.340905863791704
     ```