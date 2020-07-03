import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import datetime

# Load data
(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

# Normalize data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Model
model = tf.keras.models.load_model('cnn')

# Predict
index = 50
test_predict = model.predict(test_images)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Prediction')

# Plot predictions
ax1.bar(class_names, test_predict[index])

# Plot
plt.xticks([])
plt.yticks([])
ax2.grid(False)
ax2.imshow(test_images[index], cmap=plt.cm.binary)
plt.xlabel(class_names[test_labels[index][0]])
plt.show()
