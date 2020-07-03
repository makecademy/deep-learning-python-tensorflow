import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Read data
data = pd.read_csv('data/covtype.csv', encoding='utf-8')

# Print info
print(data.head())
print(data.shape)

# Show histogram on one column
data.hist(column='Elevation', bins=50)
plt.show()

