# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
import keras

""" 
This class functions as the base, it isn't dependent on any other local mnist related class.
Createmodel.py uses it to construct a model if one isn't saved and runmnist uses it to run a saved model.
"""

# Reoccuring variables
CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT = 28, 28
CONST_IMAGE_CHANNELS = 1
CONST_NUM_CLASSES = 10

# Import the dataset
numbers_mnist = tf.keras.datasets.mnist

# Loading the data that was imported above 
# train_images and train_labels are the training set (What the model uses to learn)
# test_images and test_labels are what the model is tested against after it's learned
(train_images, train_labels), (test_images, test_labels) = numbers_mnist.load_data()

# Manipulating the data from a 3d => 4d numpy arrays
train_images = train_images.reshape(train_images.shape[0], CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT, CONST_IMAGE_CHANNELS)
test_images = test_images.reshape(test_images.shape[0], CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT, CONST_IMAGE_CHANNELS)
input_shape = (CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT, CONST_IMAGE_CHANNELS)

# Ensuring values are floats so decimal points can be used after division
# https://stackoverflow.com/questions/48219442/use-tf-to-float-or-tf-image-convert-image-dtype-in-image-pipeline-for-cn
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# The images in the training set fall in the 0-255 value range
# Will have to scale the values to a range of 0 to 1 before feeding them to the neural network model
# To accomplish this will divide the values by 255 => Training set and testing set must be preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# Converts a class vector (integers) to binary class matrix
# https://keras.io/utils/
train_labels = keras.utils.to_categorical(train_labels, CONST_NUM_CLASSES)
test_labels = keras.utils.to_categorical(test_labels, CONST_NUM_CLASSES)



