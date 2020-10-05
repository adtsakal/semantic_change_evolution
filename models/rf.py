#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, os
import random as rn
import numpy as np
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity as cos


'''Code for the RF baseline. The "TEST_ON" variable indicates the year that
will be predicted, given the initial time point (accepted values: [1-13]). '''


def data():        
    data_folder = '../data/'

    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    train_idx = pickle.load(open(data_folder+'train_idx.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))
    trainX, testX = ts[train_idx], ts[test_idx]
    return trainX, testX


def custom_loss_func(y_test, preds):
    return np.array([cos(preds[w].reshape((1,-1)),y_test[w].reshape((1,-1)))[0, 0] for w in range(len(preds))])
   

def train_rf(x_train, y_train):
    val_start = int(0.75*len(x_train))
    x_val, y_val = x_train[val_start:len(x_train)], y_train[val_start:len(y_train)]
    x_train, y_train = x_train[0:val_start], y_train[0:val_start]
    
    trials = dict()
    for param in [50,100,150,200]:
        model = RandomForestRegressor(n_estimators=param)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        score = custom_loss_func(y_val, preds)
        print('\t\t\t', param, '\t', np.average(score))
        trials[param] = [model, np.average(score)]  
    maxval, model = -1, None
    for param in trials.keys():
        score = trials[param][1]
        if score>maxval:
            model = trials[param][0]
            maxval = score
    return model, trials


def eval_rf(x_test, y_test, model):
    preds = model.predict(x_test)
    scores = custom_loss_func(y_test, preds)
    return scores


if __name__ == '__main__':
    TEST_ON = 13
    
    train_vectors, test_vectors = data()
    x_train, y_train = train_vectors[:,0,:], train_vectors[:,TEST_ON,:]
    x_test, y_test = test_vectors[:,0,:], test_vectors[:,TEST_ON,:]
    
    model, trials = train_rf(x_train, y_train)
    results = eval_rf(x_test, y_test, model)

    folder = '../results/results_rf'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(results, open(folder+'/'+str(TEST_ON)+'.p', "wb"))