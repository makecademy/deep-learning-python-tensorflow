import numpy as np
import matplotlib.pyplot as plt

# Data
dataLength = 100
inputs = []
outputs = []

# Fixing random state for reproducibility
np.random.seed(19680801)

for i in np.arange(dataLength):
    inputs.append(i)
    outputs.append(i + 5 * np.random.rand())

# Plot
plt.scatter(inputs, outputs, s=1)
plt.show()
