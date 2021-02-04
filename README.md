# NLP-ML-for-incident-learning
This is a python program that uses Natural Language Processing (NLP) in conjunction with supervised Machine Learning (ML) techniques to semi-automate the classification of radiotherapy incident reports. In this script we compile the radiotherapy incident reports collected from the Canadian Institute for Health Information (CIHI) database as well as Safety and Incident Learning System (SaILS) database of McGill University Health Centre (MUHC) Montreal, Canada. The incident reports are processed using numerous NLP techniques. The processed reports are then used to train multiple machine learning models from the Scikit-Learn library. We extended the models to be multi-label compatible and our final model is capable of generating a drop-down menu of label suggestions to assist incident investigating personnel. Refer to the [reference](#Reference) section for more details


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
* Multi-label capabity using multi-output methods
* Extensive model evaluation
* Custom scorer
* 5-fold cross-validation
* Grid search for hyperparameter tuning

## Instructions
1. Fill-in the [MUHC datafile](0_MUHC_data.csv) and the [CIHI datafile](0_CIHI_data.csv) with the incident report data according to the templates given.
2. Run the python files in order from the [src](src) folder.
3. Obtaine the output files from the [out](out) folder.

## License
This project is provided under the MIT license. See the [LICENSE file](LICENSE) for more info.

## Reference
1.
2.
