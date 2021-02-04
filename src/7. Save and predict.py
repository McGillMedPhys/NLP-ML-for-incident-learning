#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file, the model that performed the best after hyperparameter tuning in the previous step will be fit again
with the tuned parameters to the train set and then will be saved to the system. 
The saved model will then be used to predict the labels on our test set. 
"""

import pandas
import numpy as np
import csv


from sklearn.svm.classes import LinearSVR
from sklearn.multioutput import MultiOutputRegressor

from joblib import dump, load

#### Saving the models to the system
#LinearSVR was found to be the best model for process_step, problem_type and contributing_factor datasets.

## Process Step
X_ps_train = pandas.read_csv('../out/train/X_PS_train.csv', delimiter=',', encoding='latin-1')
Y_ps_train = pandas.read_csv('../out/train/Y_PS_train.csv', delimiter=',', encoding='latin-1')

ps_model = MultiOutputRegressor(LinearSVR(C=0.2, dual=True, epsilon=0.4, fit_intercept=False, loss='squared_epsilon_insensitive', max_iter= 1000, tol= 0.01))
ps_model.fit(X_ps_train, Y_ps_train)

dump(ps_model,'../out/Process-step_Model')



## Problem type
X_pt_train = pandas.read_csv('../out/train/X_PT_train.csv', delimiter=',', encoding='latin-1')
Y_pt_train = pandas.read_csv('.../out/train/Y_PT_train.csv', delimiter=',', encoding='latin-1')

ps_model = MultiOutputRegressor(LinearSVR(C=0.2, dual=True, epsilon=0.4, fit_intercept=True, loss='squared_epsilon_insensitive', max_iter= 1000, tol= 0.01))
ps_model.fit(X_pt_train, Y_pt_train)

dump(ps_model,'../out/Problem-type_Model')


## Contributing factors
X_cf_train = pandas.read_csv('../out/train/X_CF_train.csv', delimiter=',', encoding='latin-1')
Y_cf_train = pandas.read_csv('../out/train/Y_CF_train.csv', delimiter=',', encoding='latin-1')

ps_model = MultiOutputRegressor(LinearSVR(C=0.2, dual=True, epsilon=0.4, fit_intercept=False, loss='squared_epsilon_insensitive', max_iter= 1000, tol= 0.01))
ps_model.fit(X_cf_train, Y_cf_train)

dump(ps_model,'../out/Contributing-factors_Model')





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



###### Loading the saved models for predicting on the test set
predict_score = []

#process step
X_ps_test = pandas.read_csv('../out/test/X_PS_test.csv', delimiter=',', encoding='latin-1')
Y_ps_test = pandas.read_csv('.../out/test/Y_PS_test.csv', delimiter=',', encoding='latin-1')

Process_step_Model = load('.../out/Process-step_Model')
ps_score = average_lowest_correct(Y_ps_test, Process_step_Model.predict(X_ps_test))
print ("Predicted score for Process Step is = ", ps_score)
predict_score.append({'Dataset':"Process Step",'Prediction_score':ps_score})

#problem type
X_pt_test = pandas.read_csv('.../out/test/X_PT_test.csv', delimiter=',', encoding='latin-1')
Y_pt_test = pandas.read_csv('../out/test/Y_PT_test.csv', delimiter=',', encoding='latin-1')

Problem_type_Model = load('../out/Problem-type_Model')
pt_score = average_lowest_correct(Y_pt_test, Problem_type_Model.predict(X_pt_test))
print ("Predicted score for Problem Type is = ", pt_score)
predict_score.append({'Dataset':"Problem Type",'Prediction_score':pt_score})                  

#contributing factors
X_cf_test = pandas.read_csv('../out/test/X_CF_test.csv', delimiter=',', encoding='latin-1')
Y_cf_test = pandas.read_csv('../out/test/Y_CF_test.csv', delimiter=',', encoding='latin-1')

Contributing_factor_Model = load('../out/Contributing-factors_Model')
cf_score = average_lowest_correct(Y_cf_test, Contributing_factor_Model.predict(X_cf_test))
print ("Predicted score for Contributing Factor is = ", cf_score)
predict_score.append({'Dataset':"Contributing factors",'Prediction_score':cf_score})


score = pandas.DataFrame(predict_score, columns=['Dataset','Prediction_score']) 
score.to_csv('../out/7. Prediction Scores.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
