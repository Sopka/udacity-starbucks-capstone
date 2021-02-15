import pandas as pd
import sys
from pathlib import Path

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

KERAS_MODEL_DIRECTORY = './keras_model'

# load data set
enriched_dtevent = pd.read_csv('enriched_dtevent.csv')

# convert values for income and age into 5 categories
le = LabelEncoder()
income_cats = pd.cut(enriched_dtevent.income, 5, labels=[
                     'low', 'moderate', 'considerable', 'high', 'very high'], retbins=True)[0]
enriched_dtevent['income_bins_encoded'] = le.fit_transform(income_cats)

le = LabelEncoder()
age_cuts = pd.cut(enriched_dtevent.age, 5, labels=[
                  'low', 'moderate', 'considerable', 'high', 'very high'], retbins=True)[0]
enriched_dtevent['age_bins_encoded'] = le.fit_transform(age_cuts)

# derive feature matrix and target vector for prediction
X = enriched_dtevent[['offer_type_encoded', 'gender_encoded', 'age_bins_encoded', 'income_bins_encoded',
                      'ch_social', 'ch_mobile', 'ch_email', 'ch_web', 'reward', 'difficulty', 'duration']]
y = enriched_dtevent[['completed']]


if X.isnull().any().any():
    print("Feature matrix X has null values. Exiting...")
    sys.exit(1)

print("shape of X", X.shape)

# split data set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# scale data set values between [0;1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # fit and transform
X_test = scaler.transform(X_test)  # transform


# define the keras model
model = Sequential()
model.add(Dense(128,
                input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=Adam(
    learning_rate=0.00025), metrics=['accuracy'])
# get a png of the model's layers
Path(KERAS_MODEL_DIRECTORY).mkdir(parents=True, exist_ok=True)
plot_model(model, to_file=KERAS_MODEL_DIRECTORY+'/model.png', show_shapes=True)
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150)
model.summary()
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
# print(model.metrics_names)
print('Accuracy: %.2f' % (accuracy*100))
# save model into directory 'keras_model'
model.save(KERAS_MODEL_DIRECTORY)

# make predictions with the model
predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]
print(classification_report(y_test, rounded))
