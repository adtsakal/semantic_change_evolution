#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from evaluator import get_murank, get_pk


def get_baseline_results(baseline_name, TEST_ON):
    results_folder = '../results/'
    word_scores = pickle.load(open(results_folder+'results_'+str(baseline_name)+'/'+str(TEST_ON)+'.p', 'rb'))
    inverse_order = False
    if baseline_name in ['procr', 'procr_k', 'procr_kt', 'gt_c', 'gt_beta', 'procr_star', 'procr_meanDist']:
        inverse_order = True
    murank = 100.0*get_murank(word_scores, inverse_order)
    p5 = 100.0*get_pk(word_scores, int(0.05*len(word_scores)), inverse_order)
    p10 = 100.0*get_pk(word_scores, int(0.1*len(word_scores)), inverse_order)
    p50 = 100.0*get_pk(word_scores, int(0.5*len(word_scores)), inverse_order)
    
    print('TEST_ON:', TEST_ON, 'Cosine, mu-rank, prec@k:', '\t', murank, '\t',  p5, '\t', p10, '\t', p50)
    results = [murank, p5, p10, p50]
    return results