import time
import os
from sys import argv
import csv
import pickle

import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
from keras.layers import Input
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from keras.layers.recurrent import GRU
from DatasetManager import DatasetManager
import auc_callback
from calibration_models import LSTM2D
import keras
from keras.layers import *
from keras.models import *
from sklearn.utils import class_weight
from keras.layers import Dense,Dropout,Activation,Convolution1D
# dataset_name = argv[1]
# method_name = argv[2]
# cls_method = argv[3]
# params_dir = argv[4]
# results_dir = argv[5]
from keras.layers.convolutional import *
from keras.utils.vis_utils import plot_model
from keras.layers import Input,Dense,Flatten


# dataset_name = "traffic_fines_1"
dataset_name = "bpic2012"
method_name = "BiGRU"
cls_method = "BiGRU_calibrated"
params_dir = "../parameter/BiGRU/optimal_params_BiGRU"
results_dir = "../parameter/BiGRU/result_BiGRU_autoEncoder"


optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (dataset_name, method_name, 
                                                                                       cls_method.replace("_calibrated", "")))
with open(optimal_params_filename, "rb") as fin:
    params = pickle.load(fin)
    
lstmsize = int(params['lstmsize'])
dropout = float(params['dropout'])

#n_layers = int(params['n_layers'])
n_layers = 3
batch_size = int(params['batch_size'])
optimizer = params['optimizer']
learning_rate = float(params['learning_rate'])
nb_epoch = int(params['nb_epoch'])

activation = "sigmoid"
activation_relu = "relu"

train_ratio = 0.8
val_ratio = 0.2

detailed_results_dir = "%s_detailed" % results_dir
# create results directories
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(detailed_results_dir):
    os.makedirs(detailed_results_dir)

##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, test = dataset_manager.split_data_strict(data, train_ratio)
train, val = dataset_manager.split_val(train, val_ratio)

if "traffic_fines" in dataset_name:
    max_len = 10
elif "bpic2017" in dataset_name:
    max_len = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_len = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
del data
    
dt_train = dataset_manager.encode_data_for_lstm(train)
del train
data_dim = dt_train.shape[1] - 3
X, y = dataset_manager.generate_3d_data(dt_train, max_len)
del dt_train

dt_val = dataset_manager.encode_data_for_lstm(val)
del val
X_val, y_val = dataset_manager.generate_3d_data(dt_val, max_len)
del dt_val

dt_test = dataset_manager.encode_data_for_lstm(test)
del test

print("Done: %s"%(time.time() - start))


class AutoEncoder(keras.Model):

    def __init__(self, input_shape, hidden_list, activation=activation_relu):
        super(AutoEncoder, self).__init__()

        # Encoders
        center = hidden_list.pop()
        self.encoder = Sequential([keras.layers.Dense(num, activation=activation_relu) for num in hidden_list]
                                  + [keras.layers.Dense(center)])

        # Decoders
        hidden_list.reverse()
        self.decoder = Sequential([keras.layers.Dense(num, activation=activation_relu) for num in hidden_list]
                                  + [keras.layers.Dense(input_shape)])

    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)
        return x_hat


print('Training model...')
start = time.time()
#model = Sequential()
#model = AutoEncoder(input_shape,[392,196,98,36])
#model.build(input_shape=(None, input_shape))




main_input = Input(shape=(max_len,data_dim), name='main_input')
print("main_input.shape=")
print(main_input.shape)
#ae_input=Input(shape=(None,data_dim), name='ae_input')
print(max_len)
#model.add(AutoEncoder(ae_input,[98,256,64,20]))

if n_layers == 1:
    #lstm1 = GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',return_sequences=False, dropout=dropout)
    # model.add(keras.layers.Bidirectional(lstm1,input_shape=(max_len, data_dim)))
    l1= GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',return_sequences=False, dropout=dropout)
    l2 = Bidirectional(l1,merge_mode="concat", weights=None)(main_input)
    b2_3 = BatchNormalization()(l2)
    # model.add(BatchNormalization())

elif n_layers == 2:
    #model.add(keras.layers.Bidirectional(GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',
     #                return_sequences=True, dropout=dropout),input_shape=(max_len, data_dim)))
    #model.add(BatchNormalization(axis=1))
    #model.add(keras.layers.Bidirectional(GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',
     #                return_sequences=False, dropout=dropout),merge_mode="concat", weights=None,input_shape=(max_len, data_dim)))
    #model.add(BatchNormalization())

    l1 = GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',return_sequences=True, dropout=dropout)
    l2 = Bidirectional(l1,merge_mode="concat", weights=None)(main_input)
    b1 = BatchNormalization(axis=1)(l2)
    l3 = GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)
    l4 = Bidirectional(l3,merge_mode="concat", weights=None)(b1)
    b2_3 = BatchNormalization()(l4)






elif n_layers == 3:
    #model.add(keras.layers.Bidirectional(GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',
     #                return_sequences=True, dropout=dropout),input_shape=(max_len, data_dim)))
    #model.add(BatchNormalization(axis=1))
    #model.add(keras.layers.Bidirectional(GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',
     #                return_sequences=True, dropout=dropout),input_shape=(max_len, data_dim)))
    #model.add(BatchNormalization(axis=1))
    #model.add(keras.layers.Bidirectional(GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',
     #                return_sequences=False, dropout=dropout),input_shape=(max_len, data_dim)))
    #model.add(BatchNormalization())


    l1 = GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform',return_sequences=True, dropout=dropout)


    l2 = Bidirectional(l1,merge_mode="concat", weights=None)(main_input)

    b1 = BatchNormalization(axis=1)(l2)
    print("b1.shape=")
    print(b1.shape)

    l1_2 = GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)


    l2_2 = Bidirectional(l1_2,merge_mode="concat", weights=None)(b1)
    print("l2_2.shape=")
    print(l2_2.shape)


    b2 = BatchNormalization(axis=1)(l2_2)
    print("b2.shape=")
    print(b2.shape)

    l3 = GRU(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)

    l4 = Bidirectional(l3,merge_mode="concat", weights=None)(b2)
    print("l4.shape=")
    print(l4.shape)

    b2_3 = BatchNormalization()(l4)
    print("b2_3.shape=")
    print(b2_3.shape)
#CNN

main_input1 = main_input[:,:,:,np.newaxis]
conv1 = Conv2D(16,3,strides=2,border_mode='same',init='glorot_uniform')(main_input1)
print(conv1.shape)
layer1 = Activation('relu')(conv1)
conv2 = Conv2D(32,3, strides=2,border_mode='same',)(layer1)
layer2 = Activation('relu')(conv2)
print(layer2.shape)
conv3 = Conv2D(64,3,strides=2, border_mode='same',)(layer2)
layer3 = Activation('relu')(conv3)
print(layer3.shape)
conv4 = Conv2D(1,1,strides=2, border_mode='same',)(layer3)
layer4 = Activation('relu')(conv4)
print(layer4.shape)
# m2 = MaxPooling2D()(layer4)
# print("m2.shape")
# print(m2.shape)
m2 = Flatten()(layer4)
print("Flatten之后的m2.shape")
print(m2.shape)
print("RNN得到的结果")
print(b2_3.shape)
g2 = concatenate([m2,b2_3],axis=1)

print("合并之后的shape")
print(g2.shape)


outcome_output = Dense(2, activation=activation, kernel_initializer='glorot_uniform', name='outcome_output')(g2)
print("outcome_output的shape")
print(outcome_output.shape)

model = Model(inputs=[main_input], outputs=[outcome_output])
model.summary()

plot_model(model, to_file='Flatten.png', show_shapes=True)


#model.add(Dense(2, activation=activation, kernel_initializer='glorot_uniform', name='outcome_output'))




if optimizer == "adam":
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
elif optimizer == "rmsprop":
    opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss={'outcome_output':'binary_crossentropy'}, optimizer=opt)

auc_cb = auc_callback.AUCHistory(X_val, y_val)

y_ints = [k.argmax() for k in y]
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)# 因为这里是one-hot编码，所以这里总出错
class_weight_dict = dict(enumerate(class_weights))


history = model.fit(X, y, validation_data=(X_val, y_val), verbose=2,
                    callbacks=[auc_cb], batch_size=batch_size, epochs=nb_epoch)

last_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('outcome_output').output)






#使用类别处理样本不均衡的问题
#history = model.fit(X, y, validation_data=(X_val, y_val), verbose=2,
                   #callbacks=[auc_cb], batch_size=batch_size, epochs=nb_epoch,class_weight = class_weight_dict)
if "calibrate" in cls_method:
    n_cases, time_dim, n_features = X_val.shape
    # model_2d = LSTM2D(model, time_dim, n_features)  #2020.6.10修改
    model_2d = LSTM2D(last_layer_model, time_dim, n_features)
    model_calibrated = CalibratedClassifierCV(model_2d, cv="prefit", method='sigmoid')
    model_calibrated.fit(X_val.reshape(n_cases, time_dim*n_features), y_val[:,1])

print("Done: %s"%(time.time() - start))


# Write loss for each epoch
print('Evaluating...')
start = time.time()

detailed_results = pd.DataFrame()
preds_all = []
test_y_all = []
nr_events_all = []
for nr_events in range(1, max_len+1):
    # encode only prefixes of this length
    X, y, case_ids = dataset_manager.generate_3d_data_for_prefix_length(dt_test, max_len, nr_events)

    if X.shape[0] == 0:
        break

    if "calibrate" in cls_method:
        #preds = model_calibrated.predict_proba(X.reshape(X.shape[0], time_dim*n_features))[:,1] #2020.6.10
        preds = model_calibrated.predict_proba(X.reshape(X.shape[0], time_dim * n_features))[:, 1]
    else:
        #preds = model.predict(X, verbose=0)[:,1]   #2020.6.10
        preds = last_layer_model.predict(X, verbose=0)[:,1]

    current_results = pd.DataFrame({"dataset": dataset_name, "method": method_name, "cls": cls_method,
                                    "nr_events": nr_events, "predicted": preds, "actual": y[:,1], "case_id": case_ids})
    detailed_results = pd.concat([detailed_results, current_results], axis=0)
    
    preds_all.extend(preds)
    test_y_all.extend(y[:,1])
    nr_events_all.extend([nr_events] * X.shape[0])
    
print("Done: %s"%(time.time() - start))
        
# Write results
results_file = os.path.join(results_dir, "results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
detailed_results_file = os.path.join(detailed_results_dir, "detailed_results_%s_%s_%s.csv"%(cls_method, dataset_name, method_name)) 
detailed_results.to_csv(detailed_results_file, sep=";", index=False)

with open(results_file, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
    spamwriter.writerow(["dataset", "method", "cls", "nr_events", "metric", "score"])

    dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})
    for nr_events, group in dt_results.groupby("nr_events"):
        auc = np.nan if len(set(group.actual)) < 2 else roc_auc_score(group.actual, group.predicted)
        spamwriter.writerow([dataset_name, method_name, cls_method, nr_events, -1, "auc", auc])
        print(nr_events, auc)

    auc = roc_auc_score(dt_results.actual, dt_results.predicted)
    spamwriter.writerow([dataset_name, method_name, cls_method, -1, "auc", auc])
    print(auc)