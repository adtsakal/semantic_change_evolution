#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances as distance_metric_used


'''Code for the PROCR baseline. The "TEST_ON" variable indicates the year that
will be aligned against the initial time point (accepted values: [0-13]).'''


def get_data():
    data_folder = '../data/'
    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    train_idx = pickle.load(open(data_folder+'train_idx.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))
    trainX, testX = ts[train_idx], ts[test_idx]
    return trainX, testX


def procrustes(X, Y):
    muX, muY = np.average(X, axis=0), np.average(Y, axis=0)
    X0, Y0 = X - muX, Y - muY
    normX, normY = np.linalg.norm(X0), np.linalg.norm(Y0)
    X0/=normX
    Y0/=normY

    u, w, vt = np.linalg.svd(Y0.T.dot(X0).T)
    R = u.dot(vt)
    s = w.sum()
    mtx = np.dot(Y0, R.T)*s
    err = np.sum(np.square(X0 - mtx)) #not really needed
    return  err, X0, mtx, {'muX': muX, 'muY': muY, 'normX':normX, 'normY':normY, 'R':R, 's':s}


def procrustes_all_together(x_train, y_train, x_test, y_test):
    x_all = np.concatenate((x_train, x_test))
    y_all = np.concatenate((y_train, y_test))
    _, x_test_all, y_test_all, __ = procrustes(x_all, y_all)
    
    scores = []
    for w in range(len(x_train), len(x_all)):
        cosDist = distance_metric_used(x_test_all[w].reshape((1,-1)),y_test_all[w].reshape((1,-1)))
        scores.append(cosDist[0, 0])    
    return np.array(scores)
     

def get_distances(train_vectors, test_vectors, train_until):
    results = []
    year1 = 0
    for year2 in range(1, train_until):
        x_train, y_train = train_vectors[:,year1,:], train_vectors[:,year2,:]
        x_test, y_test = test_vectors[:,year1,:], test_vectors[:,year2,:]
    
        b = procrustes_all_together(x_train, y_train, x_test, y_test)        
        results.append(b)    
    results = np.array(results)
    return np.average(results, axis=0)

    
if __name__ == '__main__':
    TEST_ON=13
    train_vectors, test_vectors = get_data()
        
    results = get_distances(train_vectors, test_vectors, TEST_ON) 
    folder = '../results/results_procr_star'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(np.array(results), open(folder+'/'+str(TEST_ON)+'.p', "wb"))