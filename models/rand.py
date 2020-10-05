#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from evaluator import get_murank, get_pk
import numpy as np
import pickle


def get_random_baseline():
    X = pickle.load(open('data/test_idx.p', 'rb'))
    for year in range(13):
        mr, r5, r10, r50 = [], [], [], []
        for run in range(10000):
            scores = np.random.randn(len(X))
            mr.append(get_murank(scores))
            r5.append(get_pk(scores, int(0.05*len(X))))
            r10.append(get_pk(scores, int(0.1*len(X))))
            r50.append(get_pk(scores, int(0.5*len(X))))
        print(year, '\t', np.average(mr), '\t', np.average(r5), '\t', np.average(r10), '\t', np.average(r50))
            