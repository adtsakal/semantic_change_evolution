#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances as distance_metric_used


'''Code for the PROCR baseline. The "TEST_ON" variable indicates the year that
will be aligned against the initial time point (accepted values: [0-13]).'''


def data():        
    data_folder = '../data/'
    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    train_idx = pickle.load(open(data_folder+'train_idx.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))
    trainX, testX = ts[train_idx], ts[test_idx]
    return trainX, testX


def procrustes(X, Y):
    #bringing everything in the same space
    muX, muY = np.average(X, axis=0), np.average(Y, axis=0)
    X0, Y0 = X - muX, Y - muY
    normX, normY = np.linalg.norm(X0), np.linalg.norm(Y0)
    X0/=normX
    Y0/=normY
    #orthogonal procrustes
    u, w, vt = np.linalg.svd(Y0.T.dot(X0).T)
    R = u.dot(vt)
    s = w.sum()
    mtx = np.dot(Y0, R.T)*s
    err = np.sum(np.square(X0 - mtx)) #not really needed
    return  err, X0, mtx, {'muX': muX, 'muY': muY, 'normX':normX, 'normY':normY, 'R':R, 's':s}


def procrustes_all_together(x_train, y_train, x_test, y_test):
    x_all = np.concatenate((x_train, x_test)) #first time step
    y_all = np.concatenate((y_train, y_test)) #second time step
    _, x_test_all, y_test_all, __ = procrustes(x_all, y_all) #getting the transformed matrices

    scores = []
    for w in range(len(x_train), len(x_all)):
        cosDist = distance_metric_used(x_test_all[w].reshape((1,-1)),y_test_all[w].reshape((1,-1)))
        scores.append(cosDist[0, 0])
    return np.array(scores)


if __name__ == '__main__':
    TEST_ON=13
    train_vectors, test_vectors = data()    
    x_train, y_train = train_vectors[:,0,:], train_vectors[:,TEST_ON,:]
    x_test, y_test = test_vectors[:,0,:], test_vectors[:,TEST_ON,:]
    
    scores = procrustes_all_together(x_train, y_train, x_test, y_test)
    folder = '../results/results_procr'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(scores, open(folder+'/'+str(TEST_ON)+'.p', 'wb'))