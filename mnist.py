# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Import the fashion dataset
numbers_mnist = tf.keras.datasets.mnist

# Train the images in the dataset
(train_images, train_labels), (test_images, test_labels) = numbers_mnist.load_data()

class_names = ['zero','one', 'two', 'three', 'four', 'five',
               'six', 'seven', 'eight', 'nine']

# Plotting the first image and showing it
'''
plt.figure()
plt.imshow(train_images[0])
plt.grid(False)
plt.show()
'''

# The images in the training set fall in the 0-255 value range
# Will have to scale the values to a range of 0 to 1 before feeding them to the neural network model
# To accomplish this will divide the values by 255 ~Training set and testing set must be preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# Verifying the data is in the correct format, displaying the first 25 images from the *training* set
# And also display their class name below for validation
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    # Getting rid of the annoying ticks coming out of each
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

model = keras.Sequential([
    # First Layer - Transforms the image formats from a 2D array (of 28 x 28px) to a 1D array (of 28 x 28px)
    # This layer just reformats the data, unstacks rows of pixels and lines them up
    keras.layers.Flatten(input_shape=(28, 28)),
    # Second Layer - Two Dense Layers. Fully connected neural layers. The first layer has 128 neurons(nodes) 
    keras.layers.Dense(128, activation='relu'),
    # 10 neuron softmax layer that returns an array of 10 probability scores that sum up to 1
    # Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes
    keras.layers.Dense(10, activation='softmax')
])

# Before the model can be trained it needs some more settings
# Optimizer -> How the model is updated based on the data it sees and its loss function.
model.compile(optimizer='adam',
# Loss function -> Measures how accurate the model is during training
              loss='sparse_categorical_crossentropy',
# Metrics -> Used to monitor the training and testing steps
              metrics=['accuracy'])

# Train the model

# Training it requires 
# 1) Feeding the model with the trained_images and trained_labels
# 2) The model learns association between the images and labels
# 3) Ask the model to make predictions about a test set. Verify the predictions match the labls from the test_labels array
model.fit(train_images, train_labels, epochs=1)

# Compare how the model performs on the test dataset
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Now with the model trained, can use it to make predictions on some images

# The model should have predicted the label for each image in the training set
predictions = model.predict(test_images)

print(predictions[0])

# Can't gauge much from that output, but the array index[7] has the highest confidence value
print("Prediction: ", np.argmax(predictions[0]))

# The model is confident this image is a 7 (class_names[7])
# Examining the test label should confirm this classification is correct

print("Actual: ", test_labels[0])


# Graphing it to look at the full set of 10 class predictions

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Looking at the 0th image, prediction and prediction array. Correct prediction labels are blue and red respectively
# A correct prediction
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

img = 'testpredict2.png'

imaghue=mpimg.imread(img)
imgplot = plt.imshow(imaghue)

plt.show()
# Plot the first X test images, their predicted labels, and the true labels.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


