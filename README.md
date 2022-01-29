# Table_CVAE_AQP

This is the code repository for the approximate query processing paper titled *'Efficient and Accurate Approximate Join Query Processing via Conditional Sample Data Generation and Size Allocation'*.

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
    git clone https://github.com/PasaLab/Table_CVAE_AQP.git
    ```

2. Prepare the raw exact query result (under `/ground_truth`) and data

3. Train the model

    * Modifying the configuration files

        * For the query configuration file,fill in the following attributes according to the query content.

            ```
            {
              ...
              "join_cols": [],
              "groupby_cols": [], 
              ...
              "sum_cols": [],
              "avg_cols": []
              ...
            }
            ```

        * For the query configuration file,turn the "train_flag" to "train" and fill in the following attributes according to the query content.

            ```
            {
              ...
              "categorical_columns": [],	// Discrete attributes involved in the query
              "numeric_columns": [],	//Contiguous attributes involved in the query
              "label_columns": [],	// Columns used for join
              ...
            }
            ```

            And fill in some model parameters.

            ```
            {
              ...
              "categorical_encoding": ,  // binary or onehot
              "numeric_encoding": ,  // mm or gaussian
              "max_clusters": , // The parameter when using gaussian encoding
              "model_type": ,
              "lr": ,
              "optimizer_type": ,
              "loss_agg_type": ,
              "gpu_num": ,
              "epochs": ,
              "inc_epochs": , 
              "batch_size": ,
              "latent_dim": ,
              "intermediate_dim": ,
              ...
            }
            ```

    * Run the `main.py`

        ```shell
        python main.py ./config/query/xxx.json
        ```

4. Execute the query

    * Turn the "train_flag" to "load" in the training configuration files.

    * Run the `main.py`

        ```shell
        python main.py ./config/query/xxx.json
        ```

## Configuration

There are two types of configuration files: query configuration files and training configuration files.

Query configuration files is under `/config/query`.An example is given below.

```shell
{
  "name": "customer_join_supplier",
  "train_config_files": [
    "./config/train/tpch_customer_torch_cvae.json", 
    "./config/train/tpch_supplier_torch_cvae.json"
  ],
  "multi_sample_times": 1,
  "operation": "aqp",
  "join_cols": ["c_nationkey","s_nationkey"],
  "groupby_cols": ["c_nationkey"], 
  "result_path": "./output/aqp_result/cs_res.csv",
  "diff_path": "./output/diff/cs_diff.csv", 
  "sum_cols": ["c_acctbal","s_acctbal"],
  "avg_cols": ["c_acctbal","s_acctbal"],
  "var": "./var/tpch-1m/cs_var.csv",
  "ground_truth": "./ground_truth/tpch-1/cs_truth.csv" // Specifies the raw exact query result
}
```

Training configuration files  is under `/config/train`.An example is given below.

```shell
{
  "name": "tpch-1-customer", 
  "data": "./datasets/tpch-1/customer.csv",  // Path to store the exported data
  "categorical_columns": [
    "c_nationkey"
  ],
  "numeric_columns": [
    "c_acctbal"
  ],
  "label_columns": [ 
    "c_nationkey"
  ],
  "categorical_encoding": "binary",  
  "numeric_encoding": "gaussian", 
  "max_clusters": 15, 
  "model_type": "torch_cvae",
  "lr": 0.001,
  "optimizer_type": "adam",
  "loss_agg_type": "mean",
  "gpu_num": 0,
  "epochs": 100,
  "inc_epochs": 100, 
  "batch_size": 512,
  "latent_dim": 150,
  "intermediate_dim": 150,
  "train_flag": "train",  
  "operation": "aqp",
  "sample_method": "statistics",
  "sample_rate": 0.1,
  "header": 1,
  "delimiter": ","
}
```

## Copyright

The code is available for research purpose only.

For commercial usage, please contact PASA Lab@Nanjing University(gurong@nju.edu.cn).