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

script_files_dir = "script_files"
output_files_dir = "output_files"

if not os.path.exists(script_files_dir):
    os.makedirs(script_files_dir)
if not os.path.exists(output_files_dir):
    os.makedirs(output_files_dir)

### Experiments with a single run ###
## RF and XGBoost ##
datasets = ["bpic2012",  "traffic_fines_1"]
n_runs = 1

method_names = ["single_agg", "prefix_index", "single_index"]
cls_methods = ["rf", "xgboost"]
results_dir = "val_results"

for dataset_name in datasets:
    
    if "bpic2017" in dataset_name or "hospital_billing" in dataset_name:
        memory = 30000
    else:
        memory = 10000
    
    for method_name in method_names:
        for cls_method in cls_methods:
            for i in range(n_random_search_iter):

                if cls_method == "rf":
                    n_estimators = np.random.randint(150, 1000)
                    max_features = loguniform(0.01, 0.9)
                    params_str = "_".join([str(n_estimators), str(max_features)])

                else:
                    n_estimators = np.random.randint(150, 1000)
                    learning_rate = np.random.uniform(0.01, 0.07)
                    subsample = np.random.uniform(0.5, 1)
                    max_depth = np.random.randint(3, 9)
                    colsample_bytree = np.random.uniform(0.5, 1)
                    min_child_weight = np.random.randint(1, 3)
                    params_str = "_".join([str(n_estimators), str(learning_rate), str(subsample), str(max_depth), 
                                           str(colsample_bytree), str(min_child_weight)])

                params = " ".join([dataset_name, method_name, cls_method, params_str, str(n_runs), results_dir])
                script_file = os.path.join(script_files_dir, "run_%s_%s_%s_%s_singlerun.sh" % (dataset_name, method_name, 
                                                                                           cls_method, params_str))
                with open(script_file, "w") as fout:
                    fout.write("#!/bin/bash\n")
                    fout.write("#SBATCH --output=%s/output_%s_%s_%s_%s_singlerun.txt" % (output_files_dir, dataset_name, method_name,
                                                                               cls_method, params_str))
                    fout.write("#SBATCH --mem=%s\n" % memory)
                    fout.write("#SBATCH --time=7-00\n")
    
                    fout.write("python experiments_param_optim_rf_xgboost.py %s" % params)

                time.sleep(1)
                subprocess.Popen(("sbatch %s" % script_file).split())
 
## LSTM ##
method_names = ["lstm"]
cls_methods = ["lstm"]
results_dir = "val_results_lstm"

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
    
                    fout.write("python experiments_param_optim_lstm.py %s" % params)

                time.sleep(1)
                subprocess.Popen(("sbatch %s" % script_file).split())
                
                
### Experiments with multiple runs ###
method_names = ["single_agg"]
cls_methods = ["rf", "xgboost"]
n_runs = 5
results_dir = "val_results_runs"

for dataset_name in datasets:
    
    if "bpic2017" in dataset_name or "hospital_billing" in dataset_name:
        memory = 30000
    else:
        memory = 10000
        
    for method_name in method_names:
        for cls_method in cls_methods:
            for i in range(n_random_search_iter):

                if cls_method == "rf":
                    n_estimators = np.random.randint(150, 1000)
                    max_features = loguniform(0.01, 0.9)
                    params_str = "_".join([str(n_estimators), str(max_features)])

                else:
                    n_estimators = np.random.randint(150, 1000)
                    learning_rate = np.random.uniform(0.01, 0.07)
                    subsample = np.random.uniform(0.5, 1)
                    max_depth = np.random.randint(3, 9)
                    colsample_bytree = np.random.uniform(0.5, 1)
                    min_child_weight = np.random.randint(1, 3)
                    params_str = "_".join([str(n_estimators), str(learning_rate), str(subsample), str(max_depth), 
                                           str(colsample_bytree), str(min_child_weight)])

                params = " ".join([dataset_name, method_name, cls_method, params_str, str(n_runs), results_dir])
                script_file = os.path.join(script_files_dir, "run_%s_%s_%s_%s_multipleruns.sh" % (dataset_name, method_name, 
                                                                                           cls_method, params_str))
                with open(script_file, "w") as fout:
                    fout.write("#!/bin/bash\n")
                    fout.write("#SBATCH --output=%s/output_%s_%s_%s_%s_multipleruns.txt" % (output_files_dir, dataset_name, 
                                                                                            method_name, cls_method, 
                                                                                            params_str))
                    fout.write("#SBATCH --mem=%s\n" % memory)
                    fout.write("#SBATCH --time=7-00\n")
    
                    fout.write("python experiments_param_optim_rf_xgboost.py %s" % params)

                time.sleep(1)
                subprocess.Popen(("sbatch %s" % script_file).split())
