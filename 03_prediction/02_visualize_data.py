import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Read data
data = pd.read_csv('data/train.csv', encoding='utf-8')

# Print info
print(data.head())

# Show histogram on one column
data.hist(column='LotArea', bins=50)
plt.show()

