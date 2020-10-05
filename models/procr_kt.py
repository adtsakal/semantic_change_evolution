#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances as distance_metric_used


'''Code for the PROCR_kt baseline. The "TEST_ON" variable indicates the year that
will be aligned against the initial time point (accepted values: [0-13]). The
"numToKeep" variable indicates the percentage of anchor words that should be
used to learn the alignment on.'''


def data():        
    data_folder = '../data/'
    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    train_idx = pickle.load(open(data_folder+'train_idx.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))
    trainX, testX = ts[train_idx], ts[test_idx]
    return trainX, testX


def find_anchors_across_time(vectors):
    distances_across_time = []
    for year1 in range(vectors.shape[1]-1): #[ {0,1}, {1,2}, ... ]
        distances = procrustes_find_distances(vectors[:,year1,:], vectors[:,year1+1,:])
        distances_across_time.append(distances)
    return np.average(np.array(distances_across_time), axis=0)


def procrustes_find_distances(X, Y): #Returns cosDist of a word in two different time periods
    _, X0, mtx, _ = procrustes(X,Y)    
    errors = []
    for w in range(len(X0)):
        e = distance_metric_used(X0[w].reshape((1,-1)),mtx[w].reshape((1,-1)))
        errors.append(e[0, 0])
    return np.array(errors)


def find_nth_smallest(dist, numToKeep):
    numToKeep = int(numToKeep*len(dist))
    threshold = np.partition(dist, numToKeep-1)[numToKeep-1]       
    indices = np.where(dist<=threshold)[0]
    return threshold, indices


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


def procrustes_transform(x, y, params):
    X0, Y0 = (x - params['muX'])/(params['normX']), (y - params['muY'])/(params['normY'])
    return X0, Y0.dot(params['R'].T)*params['s']


def procrustes_align_on_train(x_train, y_train, x_test, y_test):
    _, __, ___, params = procrustes(x_train, y_train) #what we really need is params
    x_test, y_test = procrustes_transform(x_test, y_test, params)
    scores  = []
    for w in range(len(x_test)):
        cosDist = distance_metric_used(x_test[w].reshape((1,-1)),y_test[w].reshape((1,-1)))
        scores.append(cosDist[0, 0])
    return np.array(scores)


if __name__ == '__main__':
    numToKeep = 0.5
    TEST_ON = 13

    train_vectors, test_vectors = data()
    #First find the average distance of each word with itself over time:
    distances = find_anchors_across_time(train_vectors)
    threshold, idx = find_nth_smallest(distances, numToKeep) #we will use the "idx" for the diachronic anchor words
    
    #now use them to learn a new alignment & apply the learned transformation on the test set:
    x_train, y_train = train_vectors[:,0,:], train_vectors[:,TEST_ON,:]
    x_test, y_test = test_vectors[:,0,:], test_vectors[:,TEST_ON,:]
    scores = procrustes_align_on_train(x_train[idx], y_train[idx], x_test, y_test)
    
    folder = '../results/results_procr_kt'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(scores, open(folder+'/'+str(TEST_ON)+'.p', 'wb'))