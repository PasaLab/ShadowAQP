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

nooutlier_file = 'config/train/nooutlier_sdr_flow_torch_cvae.json'
outlier_file = 'config/train/outlier_sdr_flow_torch_cvae.json'

for i in range(1,11):
    dict = get_data(nooutlier_file)
    dict['sample_rate'] = 0.01*i
    rewrite_data(nooutlier_file, dict)
    os.system('python main.py config/query/nooutlier_sdr.json >> run_logs/no'+str(i)+'-1.out 2>&1')
    os.system('python main.py config/query/nooutlier_sdr.json >> run_logs/no'+str(i)+'-2.out 2>&1')
    os.system('python main.py config/query/nooutlier_sdr.json >> run_logs/no'+str(i)+'-3.out 2>&1')

for i in range(1,11):
    dict = get_data(outlier_file)
    dict['sample_rate'] = 0.01*i
    rewrite_data(outlier_file, dict)
    os.system('python main.py config/query/outlier_sdr.json >> run_logs/'+str(i)+'-1.out 2>&1')
    os.system('python main.py config/query/outlier_sdr.json >> run_logs/'+str(i)+'-2.out 2>&1')
    os.system('python main.py config/query/outlier_sdr.json >> run_logs/'+str(i)+'-3.out 2>&1')
