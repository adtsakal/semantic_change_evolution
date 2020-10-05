#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
from evaluator import get_murank, get_pk
from sklearn.metrics.pairwise import cosine_similarity


def data(model_name, TEST_ON): #model_name:{r,f,rf}; TEST_ON: 1-13    
    data_folder = '../data/'
    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))

    X = ts[test_idx, 0:TEST_ON, :]
    if model_name=='r':
        Y = ts[test_idx, 0:TEST_ON, :]
    elif model_name=='f':
         Y = ts[test_idx, TEST_ON:, :]
    elif model_name=='rf':
        Y =  ts[test_idx, :, :]
    folder = '../results/results_seq2seq_'+model_name+'/'
    model = pickle.load(open(folder+str(TEST_ON)+'model2.p', 'rb'))
    return X, Y, model


def data_baselines(model_name, TEST_ON): #reconstruct, future; 1-13   
    data_folder = '../data/'
    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))

    X =  np.concatenate((ts[test_idx,0,:].reshape(-1,1,ts.shape[2]), ts[test_idx,TEST_ON,:].reshape(-1,1,ts.shape[2])), axis=1) 

    if model_name=='r':
        Y =  np.concatenate((ts[test_idx,0,:].reshape(-1,1,ts.shape[2]), ts[test_idx,TEST_ON,:].reshape(-1,1,ts.shape[2])), axis=1) 
    elif model_name=='f':
        X =  ts[test_idx,0,:].reshape(-1,1,ts.shape[2])
        Y = ts[test_idx, TEST_ON, :]
    
    folder = '../results/results_lstm_'+model_name+'/'
    model = pickle.load(open(folder+str(TEST_ON)+'model2.p', 'rb'))
    return X, Y, model


def get_results(model_type, model_name, TEST_ON): #{model,baseline}, {r,f,rf}, TEST_ON
    if model_type=='model':
        X, Y, model = data(model_name, TEST_ON)
    else:
        X, Y, model = data_baselines(model_name, TEST_ON)        
    predictions = model.predict(X)

    if model_name=='rf':
        Y_past = Y[:, 0:TEST_ON, :]    
        Y_future = Y[:, TEST_ON:, :]
        past_preds, future_preds = predictions[0], predictions[1]
        
        errors = []
        all_errors = []
        for timestep in range(past_preds.shape[1]):
            preds = past_preds[:,timestep,:]
            actuals = Y_past[:,timestep,:]
            tmpval = np.diag(cosine_similarity(preds, actuals))
            errors.append(tmpval)
            all_errors.append(tmpval)
        errors_past = np.array(errors)
        avg_errors_past = np.average(errors_past, axis=0)
        p5 = 100.0*get_pk(avg_errors_past, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors_past, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors_past, int(0.5*len(X)))
            
        errors = []
        for timestep in range(future_preds.shape[1]):
            preds = future_preds[:,timestep,:]
            actuals = Y_future[:,timestep,:]
            tmpval = np.diag(cosine_similarity(preds, actuals))
            errors.append(tmpval)
            all_errors.append(tmpval)
        errors_future = np.array(errors)
        avg_errors_future = np.average(errors_future, axis=0)
        p5 = 100.0*get_pk(avg_errors_future, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors_future, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors_future, int(0.5*len(X)))
        
        all_errors = np.array(all_errors)
        avg_errors_combined = np.average(all_errors, axis=0)
        murank_micro = 100.0*get_murank(avg_errors_combined)
        p5 = 100.0*get_pk(avg_errors_combined, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors_combined, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors_combined, int(0.5*len(X)))
        
        print('TEST_ON:', TEST_ON, 'Cosine mu-rank, prec@k:', '\t', murank_micro, '\t', p5, '\t', p10, '\t', p50)
        results = [murank_micro, p5, p10, p50]
    else:
        errors = []
        print('Shape of predictions:', predictions.shape)
        for timestep in range(predictions.shape[1]):
            preds = predictions[:,timestep,:]
            if (model_type=='baseline') & (model_name=='f'):
                actuals = Y
            else:
                actuals = Y[:,timestep,:] #just Y for baseline_future_lstm
            errors.append(np.diag(cosine_similarity(preds, actuals)))
        errors = np.array(errors)
        avg_errors = np.average(errors, axis=0)
        murank = 100.0*get_murank(avg_errors)
        p5 = 100.0*get_pk(avg_errors, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors, int(0.5*len(X)))
    
        print('TEST_ON:', TEST_ON, 'Cosine, mu-rank, prec@k:', '\t', murank, '\t',  p5, '\t', p10, '\t', p50)
        results = [murank, p5, p10, p50]
    return results