import yfinance as yf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

stockData = yf.Ticker("IBM")

# Get stock info
data = stockData.history(period="5y")

# Get close price
close = data['Close'].to_numpy()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

avg = moving_average(close, 50)

# Prepare
inputs = []
outputs = []

dataLength = len(avg)
windowSize = 100
lenPredict = 1

for i in np.arange(dataLength - windowSize - lenPredict - 1):
    inputData = avg[i:i+windowSize]
    outputData = avg[i+windowSize:i+windowSize+lenPredict]
    inputs.append(inputData)
    outputs.append(outputData)

inputs = np.array(inputs)
outputs = np.array(outputs)

# Plot
index = 100
futureIndex = []
for i in np.arange(lenPredict):
    futureIndex.append(windowSize + 1 + i)

plt.plot(inputs[index])

plt.scatter(futureIndex, outputs[index], c='blue')
plt.show()
