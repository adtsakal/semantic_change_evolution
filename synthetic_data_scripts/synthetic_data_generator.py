#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle, math, os


def get_data():
    data_folder = '../data/'
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))
    labels = pickle.load(open(data_folder+'labels.p', 'rb'))[test_idx]  
    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))[test_idx]
    return ts[labels!='change'], labels[labels!='change']
 
    
def generate_semantically_changing_words(W):
    V = len(W)
    words = []
    for i in range(int(0.05*V)):
        wi = np.random.randint(V)#moving word
        while wi in words:
            wi = np.random.randint(V)
        words.append(wi)
    return words 


def generate_random_pairs(W, W_i, threshold):
    from sklearn.metrics.pairwise import cosine_similarity
    
    minimum_threshold = threshold - 0.1
    V = len(W)
    targets = []
    for i in range(len(W_i)):
        wi = W_i[i]#index of moving word alpha
        vec_i = W[wi,0]
        
        wj = np.random.randint(V)#target word beta
        vec_j = W[wj,0]
        sim = cosine_similarity(vec_i.reshape((1,-1)), vec_j.reshape((1,-1)))[0,0] 
        while (wj==wi) or ((sim>threshold) or (sim<=minimum_threshold)):
            wj = np.random.randint(V)
            vec_j = W[wj,0]
            sim = cosine_similarity(vec_i.reshape((1,-1)), vec_j.reshape((1,-1)))[0,0] 
        targets.append(wj)
    return targets   


''' Sigmoidal decreasing function'''
def generate_sigmoidal_data(W_i, W_j, duration_of_func, threshold):    
    W, labels = get_data()
    if duration_of_func=='full':
        sigmata = [1.0]
        for i in range(6,-7,-1):
            sigmata.append(1/(1+math.exp(-(i*1.0))))
    elif duration_of_func=='half':
        minyear, maxyear = 4, 10
        sigmata = []
        for i in range(minyear+1):
            sigmata.append(1.0)
        for i in range(2,-3,-1):
            sigmata.append(1/(1+math.exp(-(i*1.0))))
        for i in range(14-maxyear):
            sigmata.append(0.0)
    elif duration_of_func=='ot':
        minyear, maxyear = 5, 9
        sigmata = []
        for i in range(minyear+1):
            sigmata.append(1.0)
        for i in range(1,-2,-1):
            sigmata.append(1/(1+math.exp(-(i*1.0))))
        for i in range(14-maxyear):
            sigmata.append(0.0)
    elif duration_of_func=='quarter':
        minyear, maxyear = 6, 8
        sigmata = []
        for i in range(minyear+1):
            sigmata.append(1.0)
        for i in range(0,-1,-1):
            sigmata.append(1/(1+math.exp(-(i*1.0))))
        for i in range(14-maxyear):
            sigmata.append(0.0)       
    for idx in range(len(W_i)):
        labels[W_i[idx]] = 'change'
        wi, wj = W[W_i[idx]], W[W_j[idx]]
        for year in range(14):
            sigma = sigmata[year]
            W[W_i[idx],year] = (sigma*wi[year])+((1-sigma)*wj[year])
    
    folder = '../synthetic_data/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(W, open(folder+'vectors_'+duration_of_func+'_'+str(threshold)+'.p', 'wb'))
    pickle.dump(labels, open(folder+'labels_'+duration_of_func+'_'+str(threshold)+'.p', 'wb'))
    return sigmata


if __name__ == '__main__':
    W, labels = get_data()
    W_i = generate_semantically_changing_words(W)
    
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:#threshold c
        print('Generating synthetic data for c=', threshold)
        W_j = generate_random_pairs(W, W_i, threshold)
        for duration in ['quarter', 'ot', 'half', 'full']:#duration affecting lambda
            print('\tDuration:', duration)
            generate_sigmoidal_data(W_i, W_j, duration, threshold)
    print('Synthetic data have been generated and stored in the folder synthetic_data.')