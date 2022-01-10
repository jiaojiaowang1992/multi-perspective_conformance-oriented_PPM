
import subprocess
import time
import os

import numpy as np

def loguniform(low=0, high=1):
    val = np.exp(np.random.uniform(0, 1, None))
    scaled_val = (((val - np.exp(0)) * (high - low)) / (np.exp(1) - np.exp(0))) + low
    return scaled_val

def uniform(low=0, high=1):
    val = np.random.uniform(low, high, None)
    return val

def loguniform_int(low=0, high=1):
    val = np.exp(np.random.uniform(0, 1, None))
    scaled_val = (((val - np.exp(0)) * (high - low)) / (np.exp(1) - np.exp(0))) + low
    return int(scaled_val)

n_random_search_iter = 16
# H:\Desktop\paper5\parameter\ConformanceChecking\XGBoost_5runs\script_files




script_files_dir =  "H:\Desktop\paper5\parameter\ConformanceChecking\XGBoost_5runs\script_files"
output_files_dir = "H:\Desktop\paper5\parameter\ConformanceChecking\XGBoost_5runs\output_files"

if not os.path.exists(script_files_dir):
    os.makedirs(script_files_dir)
if not os.path.exists(output_files_dir):
    os.makedirs(output_files_dir)

### Experiments with a single run ###
## RF and XGBoost ##
datasets = [
    # ["bpic2012"
          "traffic_fines_1"
     ]

## RNN ##
method_names = ["rnn"]
cls_methods = ["rnn"]
results_dir = "C:/Users/Administrator/Desktop/parameter/val_results_rnn"

n_layers_values = [1, 2, 3]
batch_size_values = [8, 16, 32, 64]
optimizer_values = ["rmsprop", "adam"]

for dataset_name in datasets:
    
    if "bpic2017" in dataset_name or "hospital_billing" in dataset_name:
        memory = 30000
    else:
        memory = 10000
        
    for method_name in method_names:
        for cls_method in cls_methods:
            for i in range(n_random_search_iter):
                lstmsize = loguniform_int(10, 150)
                dropout = uniform(0, 0.3)
                n_layers = n_layers_values[np.random.randint(0, len(n_layers_values))]
                batch_size = batch_size_values[np.random.randint(0, len(batch_size_values))]
                optimizer = optimizer_values[np.random.randint(0, len(optimizer_values))]
                learning_rate = loguniform(low=0.000001, high=0.0001)                
                params_str = "_".join([str(lstmsize), str(dropout), str(n_layers), str(batch_size), optimizer, 
                                       str(learning_rate)])

                params = " ".join([dataset_name, method_name, cls_method, params_str, results_dir])
                script_file = os.path.join(script_files_dir, "run_%s_%s_%s_%s.sh" % (dataset_name, method_name, 
                                                                                           cls_method, params_str))
                with open(script_file, "w") as fout:
                    fout.write("#!/bin/bash\n")
                    fout.write("#SBATCH --partition=gpu\n")
                    fout.write("#SBATCH --gres=gpu:1\n")
                    fout.write("#SBATCH --output=%s/output_%s_%s_%s_%s.txt" % (output_files_dir, dataset_name, method_name,
                                                                               cls_method, params_str))
                    fout.write("#SBATCH --mem=%s\n" % memory)
                    fout.write("#SBATCH --time=7-00\n")
    
                    fout.write( "python C:/Users/Administrator/Desktop/experiments/experiments_param_optim_rnn.py %s" % params)
                   
def gci(filepath):
    files = os.listdir(filepath)
    for fi in files:
        script_file = os.path.join(filepath,fi)
        #print(script_file)
        os.system(("sbatch %s" % script_file).split()[1])
        #subprocess.Popen(("sbatch %s" % script_file).split()[1], shell="True")
gci(script_files_dir)
