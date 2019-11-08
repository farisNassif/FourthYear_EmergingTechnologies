# Adapted from https://www.tensorflow.org/tutorials/keras/classification
from __future__ import absolute_import, division, print_function, unicode_literals
# Keras import and relevant modules
import keras
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
# For saving the model and weights
import simplejson as json
# Local base file
import mnistbase as mb
# Used to save the model layout to a .png file
import graphviz
from keras.utils import plot_model

""" 
This function takes no paramaters and is only ran when no model is found in the directory to
load from, creates a model and saves it locally along with the weights 
"""
def createAndSaveModel():
    """ Create the model """
    model = keras.Sequential()

    """ First CONV => POOL Layer """
    # A 2d convolution layer allows spatial convolution over images
    # Creates a convolution kernel (small matrix) 
    model.add(Conv2D(200, # 32 represents the amount of filters 
                kernel_size=(5, 5), # Kernel_size 2 integers specifying width and height of the convolution window (5x5)  
                padding='same', # Zero padding
                activation='relu', # Relu worked best with my model https://datascience.stackexchange.com/questions/18667/relu-vs-sigmoid-in-mnist-example
                input_shape=mb.input_shape)) # Passed in value is equal to (28, 28, 1), same value as that of images to pass into the model
    
    # Used for downsampling the image size
    # Eg. in a (2,2) pool it splits a pixel image into 4 chucks and takes the 4 highest values from each chunk
    model.add(MaxPooling2D(pool_size=(2, 2))) 

    """ Second CONV => POOL Layer """
    # This time add another convolution layer with slightly different paramaters 
    model.add(Conv2D(100, 
                    (3, 3), 
                    activation='relu')) 

    # Used for downsampling the image size
    model.add(MaxPooling2D(pool_size=(2, 2)))   

    # Dropout causes my model to crash when loaded, no choice but to exclude it
    # model.add(Dropout(0.2))

    """ Fully Connected Layer """
    # Flattens the 2D arrays for fully connected layers
    model.add(Flatten())  
    # Apply a dense layer with a output of 500 (nodes)
    model.add(Dense(100, 
                activation='relu'))

    model.add(Dense(50, 
                activation='relu'))

    # Apply a final dense layer with a output of 10 
    # Softmax is applied to the final layer as it can be used to represent catagorical data, outputing results ranging from 0 upto 1
    model.add(Dense(mb.CONST_NUM_CLASSES, 
                    activation='softmax'))
    # Compile the model before training
    model.compile(loss=keras.losses.categorical_crossentropy, # Loss function -> Measures how accurate the model is during training
                optimizer='adam', # Optimizer -> How the model is updated based on the data it sees and its loss function.
                metrics=['accuracy']) # Metrics -> Used to monitor the training and testing steps

    # Train the model. Training it requires..
    # 1) Feeding the model with the trained_images and trained_labels
    # 2) The model learns association between the images and labels
    # 3) Ask the model to make predictions about a test set. Verify the predictions match the labls from the test_labels array          
    model.fit(mb.train_images, mb.train_labels, # Training images and labels
              validation_data=(mb.test_images, mb.test_labels), # Evaluate the loss and any model metrics at the end of each epoch
              batch_size=100, # Too large a mini-batch size usually leads to a lower accuracy
              epochs=10, # Number of iterations
              verbose=1) # Provides a progress bar when training

    # Compare how the model performs on the test dataset
    test_loss, test_acc = model.evaluate(mb.test_images,  mb.test_labels, verbose=1)

    # This is just for my own convenience, outputting a picture of the model
    plot_model(model, to_file='model.png')

    """ Save the model """
    # Serialize to JSON
    json_file = model.to_json()
    with open("../saved_model/SavedModel.json", "w") as file:
        file.write(json.dumps(json.loads(json_file), indent=4))

    # Serialize weights to HDF5
    model.save_weights("../saved_model/SavedModelWeights.h5")
    
    # Printing for my sanity
    print("Saved model to disk")
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)