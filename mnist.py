# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# Used for predicting local image
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageOps as io
# Local file used for processing image
import imageprocessor as ip

# Reoccuring variables
CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT = 28, 28
CONST_IMAGE_CHANNELS = 1
CONST_NUM_CLASSES = 0

# Import the dataset
numbers_mnist = tf.keras.datasets.mnist

# Loading the data that was imported above 
# train_images and train_labels are the training set (What the model uses to learn)
# test_images and test_labels are what the model is tested against after it's learned
(train_images, train_labels), (test_images, test_labels) = numbers_mnist.load_data()

# Each image is mapped to a single label. Class names are NOT included with the dataset
# Must store them in this list to use later when plotting the images
class_names = ['zero','one', 'two', 'three', 'four', 'five',
               'six', 'seven', 'eight', 'nine']
CONST_NUM_CLASSES = len(class_names)

# Manipulating the data from a 3d => 4d numpy arrays
train_images = train_images.reshape(train_images.shape[0], CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT, 1)
test_images = test_images.reshape(test_images.shape[0], CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT, 1)
input_shape = (CONST_IMAGE_WIDTH, CONST_IMAGE_HEIGHT, 1)

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
train_labels[0]

# Model Creation, model variable will be used below when compiling
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) # Max pooling operation for spatial data                 
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) # Max pooling operation for spatial data  
model.add(keras.layers.Flatten())  # Flattens the 2D arrays for fully connected layers
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(CONST_NUM_CLASSES, activation='softmax')) # Introduces non-linearity in the model

model.compile(loss=keras.losses.categorical_crossentropy, # Loss function -> Measures how accurate the model is during training
              optimizer='adam', # Optimizer -> How the model is updated based on the data it sees and its loss function.
              metrics=['accuracy']) # Metrics -> Used to monitor the training and testing steps

# Train the model. Training it requires..
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

processedImage = ip.processImage("testpredict8.png")

predictions = model.predict(processedImage)
print(predictions[0])

print("Prediction: ", np.argmax(predictions[0]))