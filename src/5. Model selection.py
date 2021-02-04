#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python file evaluates the Machine learning models from the SKlearn libraries
using cross-validation method and output the test score to select top 5 models.
"""
import pandas
import csv
import numpy as np
import time
import signal
import warnings
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

# This function (taken from the web) can be used to terminate other functions that exceeds the time we give to run in seconds
def deadline(timeout, *args):
    def decorate(f):
        def handler(signum, frame):
            raise Exception

        def new_f(*args):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            return f(*args)
            signal.alarm(0)

        new_f.__name__ = f.__name__
        return new_f
    return decorate

'''
I have tested using both classifiers and regressors. Classifiers are not able to
give us more than one prediction. Therefore, a dropdown list cannot be obtained with it. 
Only regressors are hence used here. Most Regressors as base estimators do not support
multiple outputs. Therefore sklearn's multioutput meta-estimators are used to make them
support multioutput feature. 
'''
#importing all regrrssor multioutput meta-estimators
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import MultiOutputRegressor

#importing all ensembles regressor estimators
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.forest import RandomForestRegressor

#importing all base regressor estimators
from sklearn.linear_model.bayes import ARDRegression
from sklearn.linear_model.bayes import BayesianRidge
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_decomposition.cca_ import CCA
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.linear_model.coordinate_descent import ElasticNet
from sklearn.tree.tree import ExtraTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.linear_model.huber import HuberRegressor
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.semi_supervised.label_propagation import LabelPropagation
from sklearn.semi_supervised.label_propagation import LabelSpreading
from sklearn.linear_model.least_angle import Lars
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.linear_model.least_angle import LassoLars
from sklearn.linear_model.least_angle import LassoLarsIC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model.base import LinearRegression
from sklearn.svm.classes import LinearSVR
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.linear_model.coordinate_descent import MultiTaskElasticNet
from sklearn.linear_model.coordinate_descent import MultiTaskLasso
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm.classes import NuSVR
from sklearn.linear_model.omp import OrthogonalMatchingPursuit
from sklearn.cross_decomposition.pls_ import PLSCanonical
from sklearn.cross_decomposition.pls_ import PLSRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor
from sklearn.linear_model.perceptron import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model.ransac import RANSACRegressor
from sklearn.neighbors.regression import RadiusNeighborsRegressor
from sklearn.linear_model.ridge import Ridge
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.svm.classes import SVR
from sklearn.linear_model.theil_sen import TheilSenRegressor

# multioutputs = [MultiOutputRegressor, RegressorChain]
# ensembles = [AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor]
# bases = [ARDRegression, BayesianRidge, BernoulliNB, CCA, DecisionTreeRegressor, ElasticNet, ExtraTreeRegressor, GaussianNB, GaussianProcessRegressor, HuberRegressor, KNeighborsRegressor, KernelRidge, LabelPropagation, LabelSpreading, Lars, Lasso, LassoLars, LassoLarsIC, LinearDiscriminantAnalysis, LinearRegression, LinearSVR, LogisticRegression, MLPRegressor, MultiTaskElasticNet, MultiTaskLasso, MultinomialNB, NearestCentroid, NuSVR, OrthogonalMatchingPursuit, PLSCanonical, PLSRegression, PassiveAggressiveRegressor, Perceptron, QuadraticDiscriminantAnalysis, RANSACRegressor, RadiusNeighborsRegressor, Ridge, SGDRegressor, SVR, TheilSenRegressor]

multioutputs = [MultiOutputRegressor,RegressorChain]
ensembles = [AdaBoostRegressor, BaggingRegressor]
bases = [ARDRegression, BayesianRidge, BernoulliNB, CCA, DecisionTreeRegressor, ElasticNet, ExtraTreeRegressor, GaussianNB, GaussianProcessRegressor, HuberRegressor, KNeighborsRegressor, KernelRidge, LabelPropagation, LabelSpreading, Lars, Lasso, LassoLars, LassoLarsIC, LinearDiscriminantAnalysis, LinearRegression, LinearSVR, LogisticRegression, MLPRegressor, MultiTaskElasticNet, MultiTaskLasso, MultinomialNB, NearestCentroid, NuSVR, OrthogonalMatchingPursuit, PLSCanonical, PLSRegression, PassiveAggressiveRegressor, Perceptron, QuadraticDiscriminantAnalysis, RANSACRegressor, RadiusNeighborsRegressor, Ridge, SGDRegressor, SVR, TheilSenRegressor]



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


# This is the cross validate function that fits the data on all multioutput-base estimators and evaluate the model predictions based on the scorer function defined.
@deadline(180) # terminate running instance of this function if it exceeds 50 seconds
def cv_base(multioutput, base, xs, ys):
    #Here we cross_validate the model. Cross validate split the data set into train and test and return the score and time of fitting etc.
    temp = cross_validate(multioutput(base()), xs, ys, scoring=make_scorer(average_lowest_correct), n_jobs=-1, cv=5) # 5-fold cross validation
    return [multioutput.__name__, None, base.__name__, (np.sum(temp['score_time'])).round(2), np.mean(temp['test_score']).round(2)]

# This is the cross validate function that fits the data on all multioutput-ensemble estimators and evaluate the model predictions based on the scorer function defined.
@deadline(180) 
def cv_ensemble(multioutput, ensemble, xs, ys):
    temp = cross_validate(multioutput(ensemble()), xs, ys, scoring=make_scorer(average_lowest_correct), n_jobs=-1, cv=5)
    return [multioutput.__name__, ensemble.__name__, None, (np.sum(temp['score_time'])).round(2), np.mean(temp['test_score']).round(2)]

# This funtion cross-validate all the ensemble-base combinations
@deadline(180)
def cv_ensemble_base(multioutput, ensemble, base, xs, ys):
    temp = cross_validate(multioutput(ensemble(base())), xs, ys, scoring=make_scorer(average_lowest_correct), n_jobs=-1, cv=5)
    return [multioutput.__name__, ensemble.__name__, base.__name__, (np.sum(temp['score_time'])).round(2), np.mean(temp['test_score']).round(2)]


# Process step
X_ps = pandas.read_csv('../out/train/X_PS_train.csv', delimiter=',', encoding='latin-1')
Y_ps = pandas.read_csv('../out/train/Y_PS_train.csv', delimiter=',', encoding='latin-1')



ps_models = pandas.DataFrame(columns=['Multioutput', 'Ensemble', 'Base', 'Score time (s)', 'Score (lower the better)'])

row = 0
for multioutput in multioutputs:
    for base in bases:
        print ("Cross-validating the ",base.__name__, " model\n")
        try:
            results = cv_base(multioutput, base, X_ps, Y_ps)
            for i in range(5):
                ps_models.at[row, ps_models.columns[i]] = results[i]
            row += 1
        except Exception:
            pass
    print ("------ All bases are now cross-validated ------\n")
    for ensemble in ensembles:
        print ("Cross-validating the ",ensemble.__name__, " model\n")
        try:
            results = cv_ensemble(multioutput, ensemble, X_ps, Y_ps)
            for i in range(5):
                ps_models.at[row, ps_models.columns[i]] = results[i]
            row += 1
        except Exception:
            pass
    print ("-------All ensembles are now cross-validated ------\n")
    for ensemble in ensembles:
        for base in bases:
            print ("Cross-validating the ",ensemble.__name__," + ",base.__name__, "combination\n")
            try:
                results = cv_ensemble_base(multioutput, ensemble, base, X_ps, Y_ps)
                for i in range(5):
                    ps_models.at[row, ps_models.columns[i]] = results[i]
                row += 1
            except Exception:
                pass
        
ps_models.to_csv('../out/5_PS_models_evaluation.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)


# Problem type
X_pt = pandas.read_csv('../out/train/X_PT_train.csv', delimiter=',', encoding='latin-1')
Y_pt = pandas.read_csv('../out/train/Y_PT_train.csv', delimiter=',', encoding='latin-1')



pt_models = pandas.DataFrame(columns=['Multioutput', 'Ensemble', 'Base', 'Score time (s)', 'Score (lower the better)'])


row = 0
for multioutput in multioutputs:
    for base in bases:
        print ("Cross-validating the ",base.__name__, " model\n")
        try:
            results = cv_base(multioutput, base, X_pt, Y_pt)
            for i in range(5):
                pt_models.at[row, pt_models.columns[i]] = results[i]
            row += 1
        except Exception:
            pass
    print ("------ All bases are now cross-validated ------\n")
    for ensemble in ensembles:
        print ("Cross-validating the ",ensemble.__name__, " model\n")
        try:
            results = cv_ensemble(multioutput, ensemble, X_pt, Y_pt)
            for i in range(5):
                pt_models.at[row, pt_models.columns[i]] = results[i]
            row += 1
        except Exception:
            pass
    print ("-------All ensembles are now cross-validated ------\n")
    for ensemble in ensembles:
        for base in bases:
            print ("Cross-validating the ",ensemble.__name__," + ",base.__name__, "combination\n")
            try:
                results = cv_ensemble_base(multioutput, ensemble, base, X_pt, Y_pt)
                for i in range(5):
                    pt_models.at[row, pt_models.columns[i]] = results[i]
                row += 1
            except Exception:
                pass

pt_models.to_csv('../out/5_PT_models_evaluation.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)


# contributing factors
X_cf = pandas.read_csv('../out/train/X_CF_train.csv', delimiter=',', encoding='latin-1')
Y_cf = pandas.read_csv('../out/train/Y_CF_train.csv', delimiter=',', encoding='latin-1')



cf_models = pandas.DataFrame(columns=['Multioutput', 'Ensemble', 'Base', 'Score time (s)', 'Score (lower the better)'])

row = 0
for multioutput in multioutputs:
    for base in bases:
        print ("Cross-validating the ",base.__name__, " model\n")
        try:
            results = cv_base(multioutput, base, X_cf, Y_cf)
            for i in range(5):
                cf_models.at[row, cf_models.columns[i]] = results[i]
            row += 1
        except Exception:
            pass
    print ("------ All bases are now cross-validated ------\n")
    for ensemble in ensembles:
        print ("Cross-validating the ",ensemble.__name__, " model\n")
        try:
            results = cv_ensemble(multioutput, ensemble, X_cf, Y_cf)
            for i in range(5):
                cf_models.at[row, cf_models.columns[i]] = results[i]
            row += 1
        except Exception:
            pass
    print ("-------All ensembles are now cross-validated ------\n")
    for ensemble in ensembles:
        for base in bases:
            print ("Cross-validating the ",ensemble.__name__," + ",base.__name__, "combination\n")
            try:
                results = cv_ensemble_base(multioutput, ensemble, base, X_cf, Y_cf)
                for i in range(5):
                    cf_models.at[row, cf_models.columns[i]] = results[i]
                row += 1
            except Exception:
                pass
        
cf_models.to_csv('../out/5_CF_models_evaluation.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)


# # Overall Severity
# X_os = pandas.read_csv('../out/train/X_OS_train.csv', delimiter=',', encoding='latin-1')
# Y_os = pandas.read_csv('../out/train/Y_OS_train.csv', delimiter=',', encoding='latin-1')



# os_models = pandas.DataFrame(columns=['Multioutput', 'Ensemble', 'Base', 'Score time (s)', 'Score (lower the better)'])

# row = 0
# for multioutput in multioutputs:
#     for base in bases:
#         print ("Cross-validating the ",base.__name__, " model\n")
#         try:
#             results = cv_base(multioutput, base, X_os, Y_os)
#             for i in range(5):
#                 os_models.at[row, os_models.columns[i]] = results[i]
#             row += 1
#         except Exception:
#             pass
#     print ("------ All bases are now cross-validated ------\n")
#     for ensemble in ensembles:
#         print ("Cross-validating the ",ensemble.__name__, " model\n")
#         try:
#             results = cv_ensemble(multioutput, ensemble, X_os, Y_os)
#             for i in range(5):
#                 os_models.at[row, os_models.columns[i]] = results[i]
#             row += 1
#         except Exception:
#             pass
#     print ("-------All ensembles are now cross-validated ------\n")
#     for ensemble in ensembles:
#         for base in bases:
#             print ("Cross-validating the ",ensemble.__name__," + ",base.__name__, "combination\n")
#             try:
#                 results = cv_ensemble_base(multioutput, ensemble, base, X_os, Y_os)
#                 for i in range(5):
#                     os_models.at[row, os_models.columns[i]] = results[i]
#                 row += 1
#             except Exception:
#                 pass
        
# os_models.to_csv('../out/5_OS_models_evaluation.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
