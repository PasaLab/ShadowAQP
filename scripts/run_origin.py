import json
import os

def get_data(config_file):
    dict = {}
    with open(config_file, 'r') as f:
        params = json.load(f)
        dict = params
    return dict

def rewrite_data(config_file, dict):
    with open(config_file, 'w') as f:
        json.dump(dict, f, indent=4)

config_file = 'config/train/tpch_customer_with_inc_torch_cvae.json'

# origin_train
dict = get_data(config_file)
dict['inc_train_flag'] = 'origin_train'
rewrite_data(config_file, dict)

for i in range(1,11):
    dict = get_data(config_file)
    dict['sample_rate'] = 0.01*i
    rewrite_data(config_file, dict)
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/origin_train-'+str(i)+'-1.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/origin_train-'+str(i)+'-2.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/origin_train-'+str(i)+'-3.out 2>&1')

