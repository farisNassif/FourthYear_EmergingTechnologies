# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import the fashion dataset
numbers_mnist = tf.keras.datasets.mnist

# Train the images in the dataset
(train_images, train_labels), (test_images, test_labels) = numbers_mnist.load_data()


class_names = ['zero','one', 'two', 'three', 'four', 'five',
               'six', 'seven', 'eight', 'nine']

# Should return (60000, 28, 28), 60000 being the amount of images and the images being 28x28px
# print(train_images.shape)

# Each label is an integer between 1 and 9
# print (train_labels)

# Should return (10000, 28, 28), 10000 test images being 28x28px
# print(test_images.shape)

# Plotting the first image and showing it
plt.figure()
plt.imshow(train_images[0])
plt.grid(False)
plt.show()

# The images in the training set fall in the 0-255 value range
# Will have to scale the values to a range of 0 to 1 before feeding them to the neural network model
# To accomplish this will divide the values by 255 ~Training set and testing set must be preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# Verifying the data is in the correct format, displaying the first 25 images from the *training* set
# And also display their class name below for validation
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    # Getting rid of the annoying ticks coming out of each
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

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