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

# Fit
model.fit(inputs_train, outputs_train, epochs=200)

# Eval
model.evaluate(inputs_test, outputs_test, verbose=2)

# Plot
inputs_predict = np.arange(0, dataLength)
outputs_predict = model.predict(inputs_predict)

plt.plot(inputs_predict, outputs_predict, c='red')
plt.scatter(inputs, outputs, s=1)
plt.show()
