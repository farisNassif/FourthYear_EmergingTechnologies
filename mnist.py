# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import keras
# For loading the saved model
from keras.models import model_from_json
import simplejson as json
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
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


# Try load a saved model and its weights otherwise create a new one and save it
# Adapted from: https://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras
try:
  # load json and create model
  json_file = open('SavedModel.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)

  # load weights into new model
  loaded_model.load_weights("SavedModelWeights.h5")
  print("Loaded model from disk")
  # evaluate loaded model on test data 
  # Define X_test & Y_test data first
  loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
  score = loaded_model.evaluate(test_images, test_labels, verbose=0)
  print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

  processedImage = ip.processImage("testpredict2.png")
  predictions = loaded_model.predict(processedImage)
  print(predictions[0])
  print("Prediction: ", np.argmax(predictions[0]))
except:
  print("No model was found")
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
  model.fit(train_images, train_labels, epochs=10)
  # Compare how the model performs on the test dataset
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  # Save the model

  # Serialize to JSON
  json_file = model.to_json()
  with open("SavedModel.json", "w") as file:
    file.write(json.dumps(json.loads(json_file), indent=4))

  # serialize weights to HDF5
  model.save_weights("SavedModelWeights.h5")
  print("Saved model to disk")
  print('\nTest accuracy:', test_acc)