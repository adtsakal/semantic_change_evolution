#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle, os
from sklearn.metrics.pairwise import cosine_similarity


def get_murank_synthetic(scores, labels):
    z = list(zip(scores, labels))
    z.sort()
    scores, labels = zip(*z)    
    scores, labels = np.array(scores), np.array(labels)
    average_rank = np.average(np.where(labels=='change')[0])/len(scores)
    return average_rank


def data(model_name, TEST_ON, duration, threshold): #reconstruct, future, multitask; 1-13   
    synthetic_data_folder = '../synthetic_data/'
    ts = pickle.load(open(synthetic_data_folder+'vectors_'+duration+'_'+str(threshold)+'.p', 'rb'))
    labels = pickle.load(open(synthetic_data_folder+'labels_'+duration+'_'+str(threshold)+'.p', 'rb'))
    
    X = ts[:, 0:TEST_ON, :]
    if model_name=='r':
        Y = ts[:, 0:TEST_ON, :]
    elif model_name=='f':
         Y = ts[:, TEST_ON:, :]
    elif model_name=='rf':
        Y =  ts[:, :, :]
    
    folder = '../results/results_seq2seq_'+model_name+'/'
    model = pickle.load(open(folder+str(TEST_ON)+'model2.p', 'rb'))
    return X, Y, model, labels


def get_results(model_name, TEST_ON, duration, threshold):
    X, Y, model, labels = data(model_name, TEST_ON, duration, threshold)
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
            
        errors = []
        for timestep in range(future_preds.shape[1]):
            preds = future_preds[:,timestep,:]
            actuals = Y_future[:,timestep,:]
            tmpval = np.diag(cosine_similarity(preds, actuals))
            errors.append(tmpval)
            all_errors.append(tmpval)
        
        all_errors = np.array(all_errors)
        avg_errors_combined = np.average(all_errors, axis=0)
        murank_micro = get_murank_synthetic(avg_errors_combined, labels)
        
        cosines_to_return_past = np.array([np.diag(cosine_similarity(past_preds[:,i,:], Y_past[:,i,:])) for i in range(Y_past.shape[1])])
        cosines_to_return_future = np.array([np.diag(cosine_similarity(future_preds[:,i,:], Y_future[:,i,:])) for i in range(Y_future.shape[1])])
        cosines_past = np.array([np.average(np.diag(cosine_similarity(past_preds[:,i,:], Y_past[:,i,:]))) for i in range(Y_past.shape[1])])
        cosines_future = np.array([np.average(np.diag(cosine_similarity(future_preds[:,i,:], Y_future[:,i,:]))) for i in range(Y_future.shape[1])])
        results = [np.average(cosines_past), np.average(cosines_future), murank_micro, cosines_to_return_past.T[labels=='change'], cosines_to_return_future.T[labels=='change']]
    else:
        errors = []
        for timestep in range(predictions.shape[1]):
            preds = predictions[:,timestep,:]
            actuals = Y[:,timestep,:] 
            errors.append(np.diag(cosine_similarity(preds, actuals)))
        errors = np.array(errors)
        avg_errors = np.average(errors, axis=0)
        murank = get_murank_synthetic(avg_errors,labels)
    
        cosines_to_return = np.array([np.diag(cosine_similarity(predictions[:,i,:], Y[:,i,:])) for i in range(Y.shape[1])])
        cosines = np.array([np.average(np.diag(cosine_similarity(predictions[:,i,:], Y[:,i,:]))) for i in range(Y.shape[1])])
        results = [np.average(cosines), murank, cosines_to_return.T[labels=='change']]
    return results



def generate_cosine_heatmap(data, name):
    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    if name!='legend':
        if name=='sigmoid_future':
            data = np.concatenate((np.ones(data.shape[0]).reshape((-1,1)), data), axis=1)#dummy input for pseudo-perfect prediction at timestep 0 
        ax = sns.heatmap(data, vmin=-0.1, vmax=1.0, yticklabels=False, cmap="YlGnBu", cbar=False)
    else:
        ax = sns.heatmap(data, vmin=-0.1, vmax=1.0, yticklabels=False, cmap="YlGnBu", cbar=True)    
    ax.tick_params(labelsize=16)

    folder = '../img/'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    figure = ax.get_figure()    
    figure.savefig(folder+name+'2.png', dpi=400, bbox_inches='tight')
    figure.clear()
    ax.clear()
    plt.clf()


def plot_model_per_cosine(r,f,rf,duration):
    import matplotlib.pyplot as plt
    x, r_scores, f_scores, rf_scores = [], [], [], []
    for i in range(len(r)):
        x.append(i/10)
        r_scores.append(100*r[i][1])
        f_scores.append(100*f[i][1])
        rf_scores.append(100*rf[i][2])
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=19)
    ax.tick_params(axis='both', which='minor', labelsize=19)
    ax.plot(x, r_scores,  '+-', linewidth=2, color='darkgreen')
    ax.plot(x, f_scores,  'o--', linewidth=2, color='red')
    ax.plot(x, rf_scores, '*:', linewidth=2, color='blue')
    names = ['seq2seq_r','seq2seq_f','seq2seq_rf']
    
    plt.ylim([0,30])
    plt.xlabel('c', fontsize=24)
    if duration=='quarter':
        plt.legend(names, prop={'size': 24})
    
    if duration=='full':
        plt.ylabel('$\mu_r$', fontsize=30, rotation=0, labelpad=20)
    ax.grid(axis='y',linewidth=0.25, linestyle='--')


    folder = '../img/'
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    fig.savefig(folder+'results_synthetic_'+duration+".png", bbox_inches='tight', dpi=300)    
    

def get_all_results_synthetic(duration):
    results_r, results_f, results_rf = [], [], []
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        model_names = ['r', 'f', 'rf']
        for model_name in model_names:
            print(threshold, model_name)
            if model_name=='r':
                results_r.append(get_results(model_name, 14, duration, threshold))
            elif model_name=='f':
                results_f.append(get_results(model_name, 1, duration, threshold))
            else:
               results_rf.append(get_results(model_name, 7, duration, threshold))
    return results_r, results_f, results_rf