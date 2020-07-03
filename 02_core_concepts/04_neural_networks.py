import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Parameters
dataLength = 100
inputs = []
outputs = []

# Fixing random state for reproducibility
np.random.seed(19680801)

# Fill data
for i in np.arange(dataLength):
    inputs.append(i)
    outputs.append(i + 5 * np.random.rand())

# Prepare data
inputs = np.asarray(inputs)
outputs = np.asarray(outputs)

inputs = inputs / 100.
outputs = outputs / 100.

print(inputs)

inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=101)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

plt.scatter(inputs, outputs, s=1)
plt.show()
