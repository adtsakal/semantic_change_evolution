#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle


def get_ground_truth(scores, reverse):
    data_folder = '../data/'
    labels =  pickle.load(open(data_folder+'labels.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))
    labels = labels[test_idx]
    z = list(zip(scores, labels))
    z.sort()
    if reverse:
        z.reverse()
    scores, labels = zip(*z)
    return np.array(scores), np.array(labels)
    

def get_murank(scores, reverse=False):    
    scores, labels = get_ground_truth(scores, reverse)
    average_rank = np.average(np.where(labels=='change')[0]/len(scores))
    return average_rank


def get_pk(scores, k, reverse=False):
    scores, predicted = get_ground_truth(scores, reverse)
    pk = len(np.where(predicted[:k]=='change')[0])/65
    return pk