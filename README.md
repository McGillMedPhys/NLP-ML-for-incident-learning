# NLP-ML-for-incident-learning

## Table of Contents

* [Author](#author)
* [Prerequisites](#Prerequisites)
* [Features](#Features)
* [Instructions](#Instructions)
* [License](#license)


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
