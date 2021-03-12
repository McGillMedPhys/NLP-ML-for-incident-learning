# NLP-ML-for-incident-learning
This is a python program that uses Natural Language Processing (NLP) in conjunction with supervised Machine Learning (ML) techniques to semi-automate the classification of radiotherapy incident reports. In this script we compile the radiotherapy incident reports collected from the Canadian Institute for Health Information (CIHI) database as well as Safety and Incident Learning System (SaILS) database of McGill University Health Centre (MUHC) Montreal, Canada. The incident reports are processed using numerous NLP techniques. The processed reports are then used to train multiple machine learning models from the Scikit-Learn library. We extended the models to be multi-label compatible and our final model is capable of generating a drop-down menu of label suggestions to assist incident investigating personnel. Refer to the [reference](#Reference) section for more details.

[![DOI](https://zenodo.org/badge/335829173.svg)](https://zenodo.org/badge/latestdoi/335829173)

## Table of Contents

* [Author](#author)
* [Prerequisites](#Prerequisites)
* [Features](#Features)
* [Instructions](#Instructions)
* [License](#License)
* [Reference](#Reference)


## Author
Felix Mathew\
Contact email: felix.mathew@mail.mcgill.ca


## Prerequisites
- Scikit-Learn (v0.23.1)
- SpaCy (v2.3.2)
- google-trans-new (v1.1.9)
- PyEnchant (v3.1.1)


## Features:
### Natural Language Processing (NLP):
* French to English translation
* Autocorrection
* Stopword removal
* Lemmatization
* Entity replacement
### Machine Learning (ML) with Scikit-Learn:
* One-hot encoding of the class labels
* TF-IDF vectorization of the free-text data
* Multi-label capability using multi-output methods
* Extensive model evaluation
* Custom scorer
* 5-fold cross-validation
* Grid search for hyperparameter tuning

## Instructions
The Linear SVR models that we trained and tuned on our radiation oncology incident reports can be obtained from the [out](out) folder.

To develop a machine learning model on an entirely new dataset, follow the steps:
1. Fill-in the [MUHC datafile](0_MUHC_data.csv) and the [CIHI datafile](0_CIHI_data.csv) with the incident report data according to the templates given.
2. Run the python files in order from the [src](src) folder.
3. Obtaine the output files from the [out](out) folder.

## License
This project is provided under the MIT license. See the [LICENSE file](LICENSE) for more info.

## Reference
1. Angers C, Brown R, Clark B, Renaud J, Taylor R, Wilkins A. SaILS: A Free & Open Source Tool for Incident Learning. Quebec City: Canadian Organization of Medical Physicists Winter School; 2014
2. Montgomery L, Fava P, Freeman CR, Hijal T, Maietta C, Parker W, et al. Development and implementation of a radiation therapy incident learning system compatible with local workflow and a national taxonomy. J Appl Clin Med Phys. 2018;19: 259–270.
