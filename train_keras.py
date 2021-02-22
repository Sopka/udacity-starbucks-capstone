import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.metrics import Precision
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

KERAS_MODEL_DIRECTORY = './keras_model'

# load data set
enriched_dtevent = pd.read_csv('enriched_dtevent.csv')
enriched_dtevent = enriched_dtevent[enriched_dtevent.offer_type_encoded != 2]

# choose the features to predict
X = enriched_dtevent[['gender_encoded', 'age', 'income',
                      'ch_social', 'offer_type_encoded', 'difficulty', 'duration']]
y = enriched_dtevent[['completed']]


if X.isnull().any().any():
    # assertion for null values
    print("Feature matrix X has null values. Exiting...")
    sys.exit(1)

print("shape of X", X.shape)
#X_rand = pd.DataFrame(np.random.rand(**X.shape))
#X_rand.columns = X.columns

# split data set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# scale data set values between [0;1]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit and transform
X_test = scaler.transform(X_test)  # transform

# define the keras model
model = Sequential()
model.add(Dense(64,
                input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
# optimizer = optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='Adam',
              metrics=[Precision(name='score')])
# get a png of the model layers
Path(KERAS_MODEL_DIRECTORY).mkdir(parents=True, exist_ok=True)
plot_model(model, to_file=KERAS_MODEL_DIRECTORY+'/model.png', show_shapes=True)
# fit the keras model on the dataset
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_test, y_test))

model.summary()
# evaluate the keras model
_, score = model.evaluate(X_test, y_test)
# print(model.metrics_names)
print('Precision: %.2f' % (score*100))
# save model into directory 'keras_model'
model.save(KERAS_MODEL_DIRECTORY)

# make predictions with the model
predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]
print(classification_report(y_test, rounded))
print(accuracy_score(y_test, rounded))

# plot loss and accuracy scores into an image file
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['score'], label='training score')
plt.plot(history.history['val_loss'], label='test loss')
plt.plot(history.history['val_score'], label='test score')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc="upper right")
plt.savefig(KERAS_MODEL_DIRECTORY + "/keras_loss_score.png")
