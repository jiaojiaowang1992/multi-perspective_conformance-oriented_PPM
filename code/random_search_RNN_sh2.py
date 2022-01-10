
import subprocess
import time
import os
import tensorflow as tf

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# gpu usage amount
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Graph().as_default():
    sess = tf.Session(config=gpu_config)

def gci(filepath):
    files = os.listdir(filepath)
    for fi in files:
        script_file = os.path.join(filepath,fi)
        print(script_file)
        os.system(("sbatch %s" % script_file).split()[1])
        #subprocess.Popen(("sbatch %s" % script_file).split()[1], shell="True")


n_random_search_iter = 1

script_files_dir = "/home/xdw/workspace/PycharmProjects/parameter/RNN/script_files"
output_files_dir = "/home/xdw/workspace/PycharmProjects/parameter/RNN/output_files"

gci(script_files_dir)



