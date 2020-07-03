import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

# Get data
data = pd.read_csv('data/covtype.csv', encoding='utf-8')

# Clean
data.dropna(inplace=True)

# Cols
cols = list(data.columns)

# Inputs
cols.remove('Cover_Type')

data[cols] = data[cols].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

inputs = data[cols].to_numpy()

# Outputs
outputs = data['Cover_Type'].to_numpy()

# Separate training & test
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=101)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(len(cols), input_shape=[len(cols)]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(8, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# Fit
model.fit(inputs_train, outputs_train, epochs=200, batch_size=1024,
          callbacks=[tensorboard_callback], validation_split=0.1)

# Eval
model.evaluate(inputs_test, outputs_test, verbose=2)

# Save
model.save('classifier') 