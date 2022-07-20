import json
import sys
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

# tpch-cs
# config_file = 'config/query/customer_join_supplier.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.01*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cs/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cs/'+str(i)+'-2.log 2>&1')

# # tpch-cn
# config_file = 'config/query/customer_join_nation.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cn/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cn/'+str(i)+'-2.log 2>&1')

# tpcds-sw
# config_file = 'config/query/ssales_join_wsales.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.01*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpcds_sw/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpcds_sw/'+str(i)+'-2.log 2>&1')

# tpcds-ss
# config_file = 'config/query/sales_join_store.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpcds_ss/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpcds_ss/'+str(i)+'-2.log 2>&1')

# census1
# config_file = 'config/query/census.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.02*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/PSMA/census1/'+str(i)+'.log 2>&1')
#     # os.system('python main.py ' + config_file + ' >> logs/PSMA/census1/'+str(i)+'-2.log 2>&1')

# # census2
config_file = 'config/query/census2.json'
for i in range(1,11):
    dict = get_data(config_file)
    train_config_file = dict['train_config_files'][0]
    train_config_dict = get_data(train_config_file)
    train_config_dict['sample_rate'] = 0.02*i
    rewrite_data(train_config_file, train_config_dict)
    os.system('python main.py ' + config_file + ' >> lihan_logs/census2-multi/'+str(i)+'-1.log 2>&1')
    os.system('python main.py ' + config_file + ' >> lihan_logs/census2-multi/'+str(i)+'-2.log 2>&1')

# flights1
# config_file = 'config/query/flights.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.02*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/flights-multi/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/flights-multi/'+str(i)+'-2.log 2>&1')

# flights2
# config_file = 'config/query/flights2.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.02*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/flights2/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/flights2/'+str(i)+'-2.log 2>&1')

# tpch_lpp
# config_file = 'config/query/lineitem_join_partsupp_join_parts.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lpp/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lpp/'+str(i)+'-2.log 2>&1')

# tpch_lp
# config_file = 'config/query/lineitem_join_parts.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lp/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lp/'+str(i)+'-2.log 2>&1')
