# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import the fashion dataset
numbers_mnist = keras.datasets.cifar10

# Train the images in the dataset
(train_images, train_labels), (test_images, test_labels) = numbers_mnist.load_data()

# Each image is mapped to a label. Since the class names are not included with the dataset
# Gotta store them here to user later when plotting the images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']