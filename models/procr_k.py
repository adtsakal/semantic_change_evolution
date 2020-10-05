#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances as distance_metric_used


'''Code for the PROCR_k baseline. The "TEST_ON" variable indicates the year that
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


def procrustes_anchor(x_train, y_train, x_test, y_test, numToKeep):
    anchor_indices = procrustes_find_anchor(x_train, y_train, numToKeep) #first find the most static (anchor) words
    x_train_anchor = x_train[anchor_indices]
    y_train_anchor = y_train[anchor_indices]
   
    _, __, ___, params = procrustes(x_train_anchor, y_train_anchor) #we learn the transformation using the anchor words
    x_test_anchor, y_test_anchor = procrustes_transform(x_test, y_test, params) #we apply it on the test set
    
    scores = [] #finally, we measure the cosine distance on the test set
    for w in range(len(x_test_anchor)):
        cosDist = distance_metric_used(x_test_anchor[w].reshape((1,-1)),y_test_anchor[w].reshape((1,-1)))
        scores.append(cosDist[0, 0])
    return np.array(scores)

 
def procrustes_find_anchor(X, Y, numToKeep):
    _, X0, mtx, __ = procrustes(X, Y) #get the transformed matrices of the training set
    errors = [] #calculate their displacement (cosDist)
    for w in range(len(X0)):
        e = distance_metric_used(X0[w].reshape((1,-1)),mtx[w].reshape((1,-1)))
        errors.append(e[0, 0])
    errors = np.array(errors)
    threshold, indices = find_nth_smallest(errors, numToKeep) #keep the more stable ones
    return indices


def find_nth_smallest(dist, numToKeep):
    numToKeep = int(numToKeep*len(dist))
    threshold = np.partition(dist, numToKeep-1)[numToKeep-1]
    indices = np.where(dist<=threshold)[0]
    return threshold, indices


if __name__ == '__main__':
    numToKeep = 0.9
    TEST_ON = 13
    train_vectors, test_vectors = data()
    x_train, y_train = train_vectors[:,0,:], train_vectors[:,TEST_ON,:]
    x_test, y_test = test_vectors[:,0,:], test_vectors[:,TEST_ON,:]
    
    scores = procrustes_anchor(x_train, y_train, x_test, y_test, numToKeep)
    folder = '../results/results_procr_k'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(scores, open(folder+'/'+str(TEST_ON)+'.p', 'wb'))