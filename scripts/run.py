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

# inc_train
dict = get_data(config_file)
dict['inc_train_flag'] = 'inc_train'
rewrite_data(config_file, dict)

for i in range(1,11):
    dict = get_data(config_file)
    dict['sample_rate'] = 0.01*i
    rewrite_data(config_file, dict)
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/inc_train-'+str(i)+'-1.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/inc_train-'+str(i)+'-2.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/inc_train-'+str(i)+'-3.out 2>&1')

# sample_train
dict = get_data(config_file)
dict['inc_train_flag'] = 'sample_train'
rewrite_data(config_file, dict)

for i in range(1,11):
    dict = get_data(config_file)
    dict['sample_rate'] = 0.01*i
    rewrite_data(config_file, dict)
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/sample_train-'+str(i)+'-1.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/sample_train-'+str(i)+'-2.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/sample_train-'+str(i)+'-3.out 2>&1')


# old_train
dict = get_data(config_file)
dict['inc_train_flag'] = 'old_train'
rewrite_data(config_file, dict)

for i in range(1,11):
    dict = get_data(config_file)
    dict['sample_rate'] = 0.01*i
    rewrite_data(config_file, dict)
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/old_train-'+str(i)+'-1.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/old_train-'+str(i)+'-2.out 2>&1')
    os.system('python main.py config/query/customer_with_inc_join_supplier.json' + ' >> run_logs/old_train-'+str(i)+'-3.out 2>&1')
