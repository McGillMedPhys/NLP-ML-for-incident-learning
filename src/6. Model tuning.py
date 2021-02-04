#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we will fine-tune (hyperparameter tuning) top 4 models selected from the previous step using
the gridsearch method of the sklearn library
"""

import numpy as np
import pandas
import csv


from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model.ridge import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm.classes import LinearSVR
from sklearn.linear_model.stochastic_gradient import SGDRegressor


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# This is the custom scorer defined.
# The lower the score the better. A score of 1 would be the best possible score. 
def lowest_correct(trues, preds):
    num_of_options = len(trues) # number of class labels
    drop_down_options = list(reversed(np.argsort(preds))) # Based on the regressor values, highest (most probable) value first
    correct_options = [i for i in range(num_of_options) if trues[i]==1] # get the index of the correct label
    return min([drop_down_options.index(correct_option) for correct_option in correct_options]) + 1 #check how far is that index in the dropdown list and return that value
def average_lowest_correct(list_of_trues, list_of_preds):
    length = len(list_of_trues) # number of data points
    return np.mean([lowest_correct(list(list_of_trues.iloc[i]), list(list_of_preds[i])) for i in range(length)])


# Top four models selected formatted as a pipteline to be used for gridsearch
model_1 = Pipeline([('md1', MultiOutputRegressor(Ridge()))])
model_2 = Pipeline([('md2', MultiOutputRegressor(KernelRidge()))])
model_3 = Pipeline([('md3', MultiOutputRegressor(LinearSVR()))])
model_4 = Pipeline([('md4', MultiOutputRegressor(SGDRegressor()))])

# Dictionary of all the variable hyperparameters for all four models. Except of the SGD regressor, the hyperparameter list is complete. 
model_params = {
    'Multi_Ridge': {
        'model': model_1,
        'params' : {
            'md1__estimator__normalize': [True, False],
            'md1__estimator__fit_intercept': [True, False],
            'md1__estimator__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'md1__estimator__alpha': [i for i in range(10, 110, 10)],
            'md1__estimator__max_iter': [1000, 2000, 3000]}
        },
    'Multi_KernelRidge': {
        'model': model_2,
        'params': {
            'md2__estimator__alpha': [i for i in range(10, 100, 10)],
            'md2__estimator__kernel': ['linear', 'chi2', 'laplacian', 'polymonial', 'rbf'],
            'md2__estimator__degree': [2, 3, 4, 5],
            'md2__estimator__gamma': [0.1*i for i in range(0, 11, 2)]}
        },
    'Multi_LinearSVR': {
        'model': model_3,
        'params': {
            'md3__estimator__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'md3__estimator__dual': [True, False],
            'md3__estimator__fit_intercept': [True, False],
            'md3__estimator__epsilon': [0.1*i for i in range(0, 10, 2)],
            'md3__estimator__C': [0.1*i for i in range(0, 10, 2)],
            'md3__estimator__tol': [0.01*i for i in range(-2,2)],
            'md3__estimator__max_iter': [1000, 2000, 3000]}
        },
    'Multi_SGDRegressor': {
        'model': model_4,
        'params': {
            'md4__estimator__loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'md4__estimator__penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'md4__estimator__fit_intercept': [True, False],
            'md4__estimator__learning_rate': ['constant', 'optimal', 'invscaling'],
            'md4__estimator__average': [True, False],
            'md4__estimator__alpha': [0.1**i for i in range(2, 7)],
            'md4__estimator__l1_ratio': np.linspace(0.25, 1, 5),
            'md4__estimator__max_iter': [5, 50, 500,1000,2000]}
        }
    
     }




'''
The gridsearchCV function iteratively go through each permutation and combination of the given hyperparameters.
Scoring function uses greater_is_better=False argument to show that the best estimator would be the one that gives the least score.
This will cause the score to flip its sign (becomes negetive) and we can ignore this sign. The best_score function then finds the 
best score that belongs to the best parameter combinations among all combinations and output them. If we dont use the greater_is_better=False
argument in the make scorer function, the best scorer will be giving us the highest score which would correspond to the worst estimator.
'''

# Process Step
X_ps = pandas.read_csv('../out/train/X_PS_train.csv', delimiter=',', encoding='latin-1')
Y_ps = pandas.read_csv('../out/train/Y_PS_train.csv', delimiter=',', encoding='latin-1')

ps_scores = []

for model_name, mp in model_params.items():
    print ("Tuning the ",model_name, " model")
    clf_ps = GridSearchCV(estimator= mp['model'], param_grid=mp['params'], scoring=make_scorer(average_lowest_correct, greater_is_better=False), n_jobs=-1, cv=5, verbose=1)
    print ("GridSearch of the ",model_name, " model is now complete")
    clf_ps.fit(X_ps, Y_ps)
    ps_scores.append({
        'model': model_name,
        'best_score': clf_ps.best_score_*-1,
        'best_params': clf_ps.best_params_})

score_ps = pandas.DataFrame(ps_scores, columns=['model','best_score','best_params']) 
score_ps.to_csv('../out/6_PS_Tuned Models_1.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)


## Prblem type
X_pt = pandas.read_csv('../out/train/X_PT_train.csv', delimiter=',', encoding='latin-1')
Y_pt = pandas.read_csv('../out/train/Y_PT_train.csv', delimiter=',', encoding='latin-1')

pt_scores = []

for model_name, mp in model_params.items():
    print ("Tuning the ",model_name, " model")
    clf_pt = GridSearchCV(estimator= mp['model'], param_grid=mp['params'], scoring=make_scorer(average_lowest_correct, greater_is_better=False), n_jobs=-1, cv=5, verbose=1)
    print ("GridSearch of the ",model_name, " model is now complete")
    clf_pt.fit(X_pt, Y_pt)
    pt_scores.append({
        'model': model_name,
        'best_score': clf_pt.best_score_*-1,
        'best_params': clf_pt.best_params_})

score_pt = pandas.DataFrame(pt_scores, columns=['model','best_score','best_params']) 
score_pt.to_csv('../out/6_PT_Tuned Models.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)



# ## Contribution factors
X_cf = pandas.read_csv('../out/train/X_CF_train.csv', delimiter=',', encoding='latin-1')
Y_cf = pandas.read_csv('../out/train/Y_CF_train.csv', delimiter=',', encoding='latin-1')

cf_scores = []

for model_name, mp in model_params.items():
    print ("Tuning the ",model_name, " model")
    clf_cf = GridSearchCV(estimator= mp['model'], param_grid=mp['params'], scoring=make_scorer(average_lowest_correct, greater_is_better=False), n_jobs=-1, cv=5, verbose=1)
    print ("GridSearch of the ",model_name, " model is now complete")
    clf_cf.fit(X_cf, Y_cf)
    cf_scores.append({
        'model': model_name,
        'best_score': clf_cf.best_score_*-1,
        'best_params': clf_cf.best_params_})

score_cf = pandas.DataFrame(cf_scores, columns=['model','best_score','best_params']) 
score_cf.to_csv('../out/6_CF_Tuned Models.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
