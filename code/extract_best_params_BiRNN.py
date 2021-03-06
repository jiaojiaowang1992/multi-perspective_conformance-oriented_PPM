import glob
import os
import pickle
from sys import argv

import numpy as np
import pandas as pd

loss_files_dir = "../parameter/BiRNN/loss_BiRNN"
params_dir = "../parameter/BiRNN/optimal_params_BiRNN"

if not os.path.exists(params_dir):
    print("生成新的文件夹")
    os.makedirs(params_dir)

datasets = [
            "traffic_fines_1" ,
    "bpic2012"
           
            ]
method_names = ["BiRNN"]
cls_methods = ["BiRNN"]

cls_params_names = ['lstmsize', 'dropout', 'n_layers', 'batch_size', 'optimizer', 'learning_rate', 'nb_epoch']

for dataset_name in datasets:
    for method_name in method_names:
        for cls_method in cls_methods:
            files = glob.glob("%s/%s" % (loss_files_dir, "loss_%s_%s_*.csv" % (dataset_name, method_name)))
            if len(files) < 1:
                continue
            dt_all = pd.DataFrame()
            for file in files:
                dt_all = pd.concat([dt_all, pd.read_csv(file, sep=";")], axis=0, ignore_index=True)

            dt_all = dt_all[dt_all["epoch"] >= 5]
            dt_all["params"] = dt_all["params"] + "_" + dt_all["epoch"].astype(str)
            cls_params_str = dt_all["params"][np.argmin(dt_all["val_loss"])]
            print(type(cls_params_str))

            best_params = {cls_params_names[i]: val for i, val in enumerate(cls_params_str.split("_"))}
            print("打印最好的参数")
            print(best_params)
            outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (dataset_name, method_name,
                                                                                                   cls_method))
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)

