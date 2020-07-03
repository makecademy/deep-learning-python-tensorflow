# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math

# Input
x = np.linspace(-10, 10, 100)

# Sigmoid
# z = 1/(1 + np.exp(-x))

# Linear
# z = x

# Tanh
z = np.tanh(x)

# ReLU
# z = np.maximum(0, x)

# Softmax
# expo = np.exp(x)
# expo_sum = np.sum(np.exp(x))
# z = expo/expo_sum

plt.plot(x, z)
plt.xlabel("Input")
plt.ylabel("Output")

plt.show()
