#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python file will vectorize the text data into arrays of numbers that the computer can process.
Incident descriptions are vectorize following TF-IDF algorithm. 
Data labels are vectorized using the One Hot Encoding (OHE) algorithm.
Vectorized data is saved into csv files to be used for machine learning on a later stage.
"""

import csv
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

#---------------- Data vectorization ---------------#
combined = pandas.read_csv('../out/2_Preprocessed_data.csv', delimiter=',', encoding='latin-1').fillna('')

# Function that vectorizes the incident description with TF-IDF method
def return_tfidf_matrix(column, mx, mn, ng):
    tfidf = TfidfVectorizer(max_df=mx, min_df=mn, ngram_range=ng)
    X = tfidf.fit_transform(column)
    X = pandas.DataFrame.from_records(X.todense().tolist(), columns=tfidf.get_feature_names())
    return X

# Function that vectorizes the lables with One Hot Encoding (OHE)
def return_one_hot(column):
    option_set = list(set(list(column)))
    return pandas.DataFrame([[1 if value == option else 0 for option in option_set] for value in column], columns = option_set)


## Process step
only_ps = combined[combined['Process Step']!=''] # only takes the data that has Process step entry in the report, else ignore.
ps_tfidf = return_tfidf_matrix(only_ps['Processed text'], 2500, 3, (1,3)) # 2500 is the maximum frequency of the most frequent word. The minimum frequency is set to 3 is because system hangs below 3 due to the large number of data with df=1
ps_tfidf.to_csv('../out/3_PS_TFIDF.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

ps_ohe = return_one_hot(only_ps['Process Step'])
ps_ohe.to_csv('../out/3_PS_OHE.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

## Problem type
only_pt = combined[combined['Problem Type']!='']
pt_tfidf = return_tfidf_matrix(only_pt['Processed text'], 2500, 3, (1,3))
pt_tfidf.to_csv('../out/3_PT_TFIDF.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

pt_ohe = return_one_hot(only_pt['Problem Type'])
pt_ohe.to_csv('../out/3_PT_OHE.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

# ## Severity
# only_os = combined[combined['Overall Severity']!='']
# os_tfidf = return_tfidf_matrix(only_os['Processed text'], 2500, 2, (1,3))
# os_tfidf.to_csv('3_OS_TFIDF.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

# os_ohe = return_one_hot(only_os['Overall Severity'])
# os_ohe.to_csv('3_OS_OHE.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

## Contributing factors
only_cf = combined[combined['Contributing Factors']!='']
cf_tfidf = return_tfidf_matrix(only_cf['Processed text'], 2500, 3, (1,3))
cf_tfidf.to_csv('../out/3_CF_TFIDF.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

# The contributing factor labels are dealt seperately as a single report can have multiple labels in it seperated by a '|' symbol.
column = only_cf['Contributing Factors']
option_set = set([cf for cfs in column for cf in cfs.split('|')])
cf_ohe = pandas.DataFrame([[1 if option in value else 0 for option in option_set] for value in column], columns = option_set)
cf_ohe.to_csv('../out/3_CF_OHE.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
#print ("It is complete")
print ("No.of data in PS = ",len(only_ps),"\nNo.of data in PT = ", len(only_pt),"\nNo.of data in CF = ",len(only_cf))
