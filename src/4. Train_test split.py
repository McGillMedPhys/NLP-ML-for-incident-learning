#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python file splits the preprocessed and vectorized data into training and testing subsets and stores seperately.
"""

import pandas
import csv
from sklearn.model_selection import train_test_split


## Process step
# reading data
X_ps = pandas.read_csv('../out/3_PS_TFIDF.csv', delimiter=',', encoding='latin-1')
Y_ps = pandas.read_csv('../out/3_PS_OHE.csv', delimiter=',', encoding='latin-1')
# splitting the data into train and test sets at ratio (80% : 20%) and saving them into seperate csv files
x_ps_train, x_ps_test, y_ps_train, y_ps_test = train_test_split(X_ps, Y_ps, test_size=0.20)
x_ps_train.to_csv('../out/train/X_PS_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
y_ps_train.to_csv('../out/train/Y_PS_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
x_ps_test.to_csv('../out/test/X_PS_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
y_ps_test.to_csv('../out/test/Y_PS_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)


## Prbblem type
# reading data
X_pt = pandas.read_csv('../out/3_PT_TFIDF.csv', delimiter=',', encoding='latin-1')
Y_pt = pandas.read_csv('../out/3_PT_OHE.csv', delimiter=',', encoding='latin-1')
# splitting the data into train and test sets at ratio (80% : 20%) and saving them into seperate csv files
x_pt_train, x_pt_test, y_pt_train, y_pt_test = train_test_split(X_pt, Y_pt, test_size=0.20)
x_pt_train.to_csv('../out/train/X_PT_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
y_pt_train.to_csv('../out/train/Y_PT_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
x_pt_test.to_csv('../out/test/X_PT_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
y_pt_test.to_csv('../out/test/Y_PT_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)


## Contribution factors
# reading data
X_cf = pandas.read_csv('../out/3_CF_TFIDF.csv', delimiter=',', encoding='latin-1')
Y_cf = pandas.read_csv('../out/3_CF_OHE.csv', delimiter=',', encoding='latin-1')
# splitting the data into train and test sets at ratio (80% : 20%) and saving them into seperate csv files
x_cf_train, x_cf_test, y_cf_train, y_cf_test = train_test_split(X_cf, Y_cf, test_size=0.20)
x_cf_train.to_csv('../out/train/X_CF_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
y_cf_train.to_csv('../out/train/Y_CF_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
x_cf_test.to_csv('../out/test/X_CF_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
y_cf_test.to_csv('../out/test/Y_CF_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)


# ## Overall severity
# # reading data
# X_os = pandas.read_csv('3_OS_TFIDF.csv', delimiter=',', encoding='latin-1')
# Y_os = pandas.read_csv('3_OS_OHE.csv', delimiter=',', encoding='latin-1')
# # splitting the data into train and test sets at ratio (80% : 20%) and saving them into seperate csv files
# x_os_train, x_os_test, y_os_train, y_os_test = train_test_split(X_os, Y_os, test_size=0.20)
# x_os_train.to_csv('./Machine_Learning/train/X_OS_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
# y_os_train.to_csv('./Machine_Learning/train/Y_OS_train.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
# x_os_test.to_csv('./Machine_Learning/test/X_OS_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
# y_os_test.to_csv('./Machine_Learning/test/Y_OS_test.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
