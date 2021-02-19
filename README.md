
# Starbucks Capstone Challenge

## Introduction

This repository deals with the _Starbucks Capstone Challenge_ offered
by udacity's datas science nanodegree program.

The `data` directory contains a simulated data set that mimics customer
behavior on the Starbucks rewards mobile app.

The goal is to derive a predictive model from the data to suggest if
there is a high chance that a customer will make a purchase as a
response to a discount or 'by one get one free' offer.

For more information read the corresponding blog post on medium: <https://medium.com/@sopka/prediction-model-for-starbucks-bogo-and-discount-offers-e63383fd6469>

## File Structure

`├── data_aggregation.ipynb` jupyter notebook for the data analysis and preparation

`├── train_keras.py` python script to train keras neural network model

`├── train_model.ipynb` jupyter notebook to fit a linear regression model

`├── data`

`│   ├── profile.json`
Rewards program users (17000 users x 5 fields)
 - gender: (categorical) M, F, O, or null 
 - age: (numeric) missing value encoded as 118 
 - id: (string/hash) 
 - became_member_on: (date) format YYYYMMDD 
 - income: (numeric) 

`│   ├── portfolio.json` 
Offers sent during 30-day test period (10 offers x 6 fields)
 - reward: (numeric) money awarded for the amount spent 
 - channels: (list) web, email, mobile, social 
 - difficulty: (numeric) money required to be spent to receive reward 
 - duration: (numeric) time for offer to be open, in days 
 - offer_type: (string) bogo, discount, informational 
 - id: (string/hash) 

`│   ├── transcript.json`
Event log (306648 events x 4 fields)
- person: (string/hash) 
- event: (string) offer received, offer viewed, transaction, offer completed 
- value: (dictionary) different values depending on event type
    - offer id: (string/hash) not associated with any "transaction" 
    - amount: (numeric) money spent in "transaction" 
    - reward: (numeric) money gained from "offer completed" 
- time: (numeric) hours after start of test 


## Prerequisites

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

Now you can install the needed python libraries:

```sh
pip install --upgrade pip
pip install pylint
pip install autopep8
pip install pandas
pip install jupyterlab
pip install matplotlib
pip install plotly
pip install scikit-learn
pip install seaborn
pip install SQLAlchemy
pip install tensorflow
pip install keras
pip install pydot
```
