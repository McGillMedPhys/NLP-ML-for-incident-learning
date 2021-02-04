#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file will do the basic text processing of the incident descriptions.
This includes line-break removal, French-->English translation, Spelling correction, punctuation removal, entity replacements, lemmatization and stopword removal.
"""

import csv
import pandas
from googletrans import Translator # This is the original google translator API. But it has some issues that needs to be fixed by google
from google_trans_new import google_translator # This is an alternative that solves the issue
import enchant
import re
import spacy


#---------------- Basic text processing ---------------#
combined = pandas.read_csv('1_Combined.csv', delimiter=',', encoding='latin-1').fillna('')  # reading the file
#combined = pandas.read_csv('test.csv', delimiter=',', encoding='latin-1').fillna('')  # reading the file


# line-break removal
combined['Incident Description'] = combined['Incident Description'].apply(lambda x:x.replace('\r', '.').replace('\n', '.')) #this removes any line-breaks in the indcident description


# English translation
# Detect whether incident descriptions are in English or French and translate if in French
translator = google_translator() # Note that if we use the original google translator API, the language detection and tranlation functions will be slightly different.
def translate (string):
    if translator.detect(string) == 'en':
        return string
    else:
        return translator.translate(string)
    
#This function replaces words with their concepts.
def replace_entities(string):  
    ret = string
    for ent in nlp(string).ents:
        if ent.label_ in ['TIME', 'DATE', 'PERCENT', 'QUANTITY', 'ORDINAL']:
            ret = ret.replace(ent.text, ' '+ent.label_.lower()+' ')
    return ret

# This function removes everything but the alphabets (including numbers and all punctuations)
def remove_punctNnum(string):
    return re.sub(r'[^A-z]', ' ', string.lower()) # change everything to lower case

# Auto-correct
# Only autocorrect words with more than 4 characters to avoid autocorrecting abbreviations which are usually <5 characters.
# Common abbreviation will be detected and kept, but not so common ones will get replaced. 
ignore_list=["vacloc","headrest","brachy","brachytherapy","isoshift"] # some words I found the autocorrect is doing a bad job on. 
d = enchant.Dict("en_US")
def autocorrect(string):
    words = []
    for token in string.split(' '):
        if len(token)>4 and not d.check(token): # d.check checks if the spelling is correct 
            if token == 'isocentre' or token == 'isocenter': # This word is treated differently because the word do not exist in the dictionary used here.
                words.append('isocentre')
            elif token in ignore_list: # if the word is in the ignore word list, then do not perform autocorrection. 
                words.append(token)
            else:
                try:
                    words.append(d.suggest(token)[0]) # d.suggest gives the correction recommendations
                except Exception:
                    words.append(token)
        else:
            words.append(token)
    return ' '.join([word for word in words if len(word)>1]) # if there are words with one character, remove it. (This removes any mistakes left after punctuation removal)

# This function removes whitespace if there are more than one whitespace between words.
def remove_whitespace(string):
    return re.sub(' +', ' ',string)

# This function converts words to their lemma if it is not a stop word and returns the lemma (essentially removes the stop word and lemmatize)
nlp = spacy.load('en')
def stopWordRemove_lemmatize(string):
    lems = [token.lemma_ for token in nlp(string) if not token.is_stop]
    return ' '.join(lems)

## This is the single function that combines all the functions defined above. A single function is defined to save computation time.
# The order in which these functions and arranged in this one functiion is crucial for the perfect functioning/ error-free text processing.
def allinOne(string):
    return (stopWordRemove_lemmatize(remove_whitespace(autocorrect(remove_punctNnum(replace_entities(translate(string)))))))


combined['Processed text'] = ''
combined['Processed text'] = combined['Incident Description'].apply(allinOne)


combined.to_csv('2_Preprocessed_data.csv', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

