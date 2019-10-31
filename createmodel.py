# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
import keras
# For saving the model and weights
import simplejson as json
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# Local base file
import mnistbase as mb

def createAndSaveModel():
    # Model Creation, model variable will be used below when compiling
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu', input_shape=mb.input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) # Max pooling operation for spatial data                 
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) # Max pooling operation for spatial data  
    model.add(keras.layers.Flatten())  # Flattens the 2D arrays for fully connected layers
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(mb.CONST_NUM_CLASSES, activation='softmax')) # Introduces non-linearity in the model

    model.compile(loss=keras.losses.categorical_crossentropy, # Loss function -> Measures how accurate the model is during training
                optimizer='adam', # Optimizer -> How the model is updated based on the data it sees and its loss function.
                metrics=['accuracy']) # Metrics -> Used to monitor the training and testing steps

    # Train the model. Training it requires..
    # 1) Feeding the model with the trained_images and trained_labels
    # 2) The model learns association between the images and labels
    # 3) Ask the model to make predictions about a test set. Verify the predictions match the labls from the test_labels array          
    model.fit(mb.train_images, mb.train_labels, epochs=14)
    # Compare how the model performs on the test dataset
    test_loss, test_acc = model.evaluate(mb.test_images,  mb.test_labels, verbose=2)

    # Save the model
    # Serialize to JSON
    json_file = model.to_json()
    with open("SavedModel/SavedModel.json", "w") as file:
        file.write(json.dumps(json.loads(json_file), indent=4))

    # serialize weights to HDF5
    model.save_weights("SavedModel/SavedModelWeights.h5")
    print("Saved model to disk")
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)