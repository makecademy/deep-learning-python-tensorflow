import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

# Get data
data = pd.read_csv('data/train.csv', encoding='utf-8')
data = data.drop(['Id', 'Alley'], axis=1)

data_categoric = data.select_dtypes(include=['object'])
data_categoric = pd.get_dummies(data_categoric)

data_numeric = data.select_dtypes(exclude=['object'])

# Clean
data_numeric.dropna(inplace=True)
data_categoric.dropna(inplace=True)

# Normalise
cols_to_norm = list(data_numeric.columns)

data_numeric[cols_to_norm] = data_numeric[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

data = data_numeric.merge(data_categoric, left_index = True, right_index = True)

# Cols
cols = list(data.columns)

# Inputs
cols.remove('SalePrice')

inputs = data[cols].to_numpy()

# Outputs
outputs = data['SalePrice'].to_numpy()

# Separate training & test
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=101)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(len(cols), input_shape=[len(cols)]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# Fit
model.fit(inputs_train, outputs_train, epochs=1500,
          callbacks=[tensorboard_callback])

# Eval
model.evaluate(inputs_test, outputs_test, verbose=2)

# Save
model.save('predict_data') 