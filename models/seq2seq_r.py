#!/usr/bin/env python3
# -*- coding: utf-8 -*-
seed_value= 0
import os, random, pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, RepeatVector
from hyperopt import tpe, STATUS_OK, Trials
from hyperas import optim
from sklearn.metrics.pairwise import cosine_similarity


'''Code for the seq2seq_r model. The "TEST_ON" variable indicates the year 
following the last year to be reconstructed; e.g., if TEST_ON=6, then the years 
[0,...,5] will be reconstructed (accepted values: [2-14]).'''


def data():
    TEST_ON = 14
    data_folder = '../data/'

    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    train_idx = pickle.load(open(data_folder+'train_idx.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))

    trainX, trainY = ts[train_idx, 0:TEST_ON, :], ts[train_idx, 0:TEST_ON, :] #target in reverse order
    testX, testY = ts[test_idx, 0:TEST_ON, :], ts[test_idx, 0:TEST_ON, :] #again

    return trainX, trainY, testX, testY


def create_lstm_model(trainX, trainY, testX, testY):
    model = Sequential()
    
    model.add(LSTM({{choice([128,256,512])}}, input_shape=(trainX.shape[1],100), return_sequences=True))
    model.add(TimeDistributed(Dropout({{choice([0.1,0.25,0.5])}})))
    model.add(LSTM({{choice([32,64])}}, go_backwards=False))
    model.add((Dropout({{choice([0.1,0.25,0.5])}})))
    
    model.add((RepeatVector(trainX.shape[1])))
    
    model.add(LSTM({{choice([32,64])}},return_sequences=True))
    model.add(TimeDistributed(Dropout({{choice([0.1,0.25,0.5])}})))
    model.add(LSTM({{choice([128,256,512])}}, go_backwards=False, return_sequences=True))
    model.add(TimeDistributed(Dropout({{choice([0.1,0.25,0.5])}})))

    model.add(TimeDistributed(Dense(100)))
    
    val_start = int(0.75*len(trainX))
    
    model.compile(loss="mean_squared_error",  optimizer="adam")
    result = model.fit(trainX[0:val_start], trainY[0:val_start],
              batch_size={{choice([32,64,128,256,512,1024])}},
              epochs={{choice([10,20,30,40,50])}},
              verbose=2)
    val_true  = trainY[val_start:len(trainX)]
    preds = model.predict(trainX[val_start:len(trainX)])
    cosines = np.array([np.average(np.diag(cosine_similarity(preds[:,i,:], val_true[:,i,:]))) for i in range(trainX.shape[1])])
    val_loss = np.average(cosines)
    
    print('\n',model.summary(), '\t\tBest val cosine:', val_loss, '\n')
    return {'loss': -val_loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    TEST_ON = 14
    trials = Trials()
    best_run, best_model = optim.minimize(model=create_lstm_model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=25,
                                              eval_space=True,
                                              trials=trials)
    
    folder = '../results/results_seq2seq_r'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(trials, open(folder+'/'+str(TEST_ON)+'trials2.p', "wb"))
    pickle.dump(best_run, open(folder+'/'+str(TEST_ON)+'params2.p', "wb"))
    pickle.dump(best_model, open(folder+'/'+str(TEST_ON)+'model2.p', "wb"))