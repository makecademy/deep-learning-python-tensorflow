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
model = tf.keras.models.load_model('classifier')

# Predict
train_predict = model.predict(inputs_train)
test_predict = model.predict(inputs_test)

print(outputs_test[25])
plt.bar(np.arange(8), test_predict[25])

plt.show()
