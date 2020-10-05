#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances as distance_metric_used
from sklearn.linear_model import LinearRegression


'''
A large body of this was taken from https://github.com/alan-turing-institute/room2glo
and the EMNLP paper "Room to Glo: A systematic comparison of semantic change 
detection approaches with word embeddings"
Many thanks to Philippa Shoemark for her clarifications!
'''


def get_data():    
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


#procrustes aligning everything together
def procrustes_all_together(x_train, y_train, x_test, y_test):
    x_all = np.concatenate((x_train, x_test))
    y_all = np.concatenate((y_train, y_test))
    _, x_test_all, y_test_all, __ = procrustes(x_all, y_all)
    scores = []
    for w in range(len(x_train), len(x_all)):
        scores.append(distance_metric_used(x_test_all[w].reshape((1,-1)),y_test_all[w].reshape((1,-1)))[0, 0])
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
    dist_dict = dict()
    for i in range(len(results)): #for every year
        tmp = results[i]
        inner_dict = dict()
        for word_idx in range(len(tmp)):
            inner_dict[word_idx] = tmp[word_idx]
        dist_dict[i] = inner_dict
    return dist_dict

    
def convert_dist_dict(dist_dict):
    """
    Dictionaries of distances  are keyed first by time-slice, then by word.
    This function converts them s.t. they are keyed first by word, then by time-slice.
    """
    dist_dict2 = {}
    for time_slice in dist_dict:
        for word in range(len(dist_dict[time_slice])): #words procesed sequentially
            if word in dist_dict2:
                dist_dict2[word][time_slice] = dist_dict[time_slice][word]
            else:
                dist_dict2[word] = {}
                dist_dict2[word][time_slice] = dist_dict[time_slice][word]
    return dist_dict2

    
if __name__ == '__main__':
    TEST_ON = 14    
    train_vectors, test_vectors = get_data()
        
    #calculate distances
    dict_of_dist_dicts = get_distances(train_vectors, test_vectors, TEST_ON) #{Year->{word->cosDist}}
    time_slices_used = sorted(list(dict_of_dist_dicts.keys())) #[1,2,3,...]    
    data = convert_dist_dict(dict_of_dist_dicts) #{word->{year->cosDist}}

    results = []
    for word in range(len(data)): #words are indexed this way
        struct_time_cosine = data[word]
        cos_dist =list(struct_time_cosine.values())
        regression_model = LinearRegression()
        regression_model.fit(np.asarray(list(range(len(cos_dist)))).reshape(-1,1), np.asarray(cos_dist).reshape(-1, 1))
        beta=regression_model.coef_[0][0]
        results.append(np.absolute(beta))
    
    folder = '../results/results_gt_beta'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(results, open(folder+'/'+str(TEST_ON)+'.p', "wb"))