
# Starbucks Capstone Challenge

This repository deals with the _Starbucks Capstone Challenge_ offered
by udacity's datas science nanodegree program.

Included is a data set form Starbucks with simulated data containing
transaction logs for persons interacting with offers distributed by Starbucks 
to end customers.

## Project Motivation

The goal is to derive a predictive model from the data to suggest if
there is a high chance that a person will spend money as a response 
to a discount offer or ‘by one get one free’ offer.

**For more information read the corresponding blog post on medium**: <https://medium.com/@sopka/prediction-model-for-starbucks-bogo-and-discount-offers-e63383fd6469>

## Files Structure

`├── data_aggregation.ipynb` jupyter notebook for the data exploration, processing and aggregation and statistical evaluation

`├── train_keras.py` python script to train keras neural network model

`├── train_model.ipynb` jupyter notebook to construct and evaluate feature matrix and fit models with linear regression and radius near neighbors classification

`├── data` This directory contains a simulated data set that mimics customer behavior on the Starbucks rewards mobile app.

`│   ├── profile.json` Represents fictive or anonymized Rewards Program Users (17000 users x 5 fields)
 - gender: (categorical) M, F, O, or null 
 - age: (numeric) missing value encoded as 118 
 - id: (string/hash) 
 - became_member_on: (date) format YYYYMMDD 
 - income: (numeric) 

`│   ├── portfolio.json` Defines offers sent during 30-day test period (10 offers x 6 fields)
 - reward: (numeric) money awarded for the amount spent 
 - channels: (list) web, email, mobile, social 
 - difficulty: (numeric) money required to be spent to receive reward 
 - duration: (numeric) time for offer to be open, in days 
 - offer_type: (string) bogo, discount, informational 
 - id: (string/hash) 

`│   ├── transcript.json` Contains event logs (306648 events x 4 fields)
- person: (string/hash) 
- event: (string) offer received, offer viewed, transaction, offer completed 
- value: (dictionary) different values depending on event type
    - offer id: (string/hash) not associated with any "transaction" 
    - amount: (numeric) money spent in "transaction" 
    - reward: (numeric) money gained from "offer completed" 
- time: (numeric) hours after start of test 


## Installation

All scripts and python source codes are tested under osx and linux only.

### Initialize the python environment

You do not want to polute your local environment with new
python libraries. Instead you should create a virtual python environment
in the current directory. The only requirement is that python3.8 is
preinstalled on your local system. Then just run:

```sh
python3 -m venv venv
```

And activate the virtual environment in your current shell:

```sh
source ./venv/bin/activate
```

### Install necessary Libraries

We are using the following libraries:

* pandas - Python Data Analysis Library <https://pandas.pydata.org/>
* numpy - The fundamental package for scientific computing with Python <https://numpy.org/>
* matplotlib - Visualization with Python <https://matplotlib.org/stable/index.html>
* plotly - Plotly Python Open Source Graphing Library <https://plotly.com/python/>
* scikit-learn - Machine Learning in Python <https://scikit-learn.org>
* seaborn - Statistical data visualization<https://seaborn.pydata.org/>
* keras - Built on top of TensorFlow 2.0, Keras is deep learning framework <https://keras.io/>
 * tensorflow - An end-to-end open source machine learning platform <https://www.tensorflow.org/> 

You can install the needed python libraries with pip in your `venv` environment:

```sh
pip install --upgrade pip
pip install pandas
pip install jupyterlab
pip install matplotlib
pip install plotly
pip install scikit-learn
pip install seaborn
pip install tensorflow
pip install keras
```

## Project Analysis

Start with the jupyter notebook `data_aggregation.ipynb`.
A complete run will aggregate the data into a CVS file in root of this directory called 
`enriched_dtevent.csv`. 

This file is then used in the jupyter notebook `train_model.ipynb` to train models with linear regression and radius near neighbors classification.

The accuracy of these model will be around 65%.

There is also a python script called `train_keras.py`. 
It will also use the aggregated data from the CVS file `enriched_dtevent.csv` to train a simple neural network with keras.

The accurcy of this model will also be around 65%.

## Acknowledgements

Thank you Starbucks and Udacity for providing the data sets!