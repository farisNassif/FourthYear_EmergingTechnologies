# Keras import
import keras
# For loading the saved model
from keras.models import model_from_json
import simplejson as json
# Helper libraries
import numpy as np
# Local file used for processing image
import imageprocessor as ip
# Local file used as dependency
import mnistbase as mb
# Local file used to create model
import createmodel as cm
# Local file used for various QoL functions
import utilities as ut


"""
This class functions as a runner for the model, ran by the main runner, runner.py.
The class passes an image from the runner to the image processer, after the image is processed it's sent off to be predicted.

If no model is found (Saved) then a new one will be created and saved, the program will then run as normal.
"""

def predict(image):
  # Taking the canvas image and converting it to an array for pre-processing (This is passed as a param to preprocess)
  # Then process, resize, flatten the image so that it resembels an MNIST dataset image
  processedImage = ip.preprocess_image(ut.image_to_array(image))
  # Pass the processed image to the model
  prediction = loaded_model.predict(processedImage)
  # Return the prediction
  return np.argmax(prediction[0])

# Try load a saved model and its weights otherwise create a new one and save it
# Adapted from: https://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras
try:
  # load json and create model
  json_file = open('../saved_model/SavedModel.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)

  # load weights into new model
  loaded_model.load_weights("../saved_model/SavedModelWeights.h5")
  print("Loaded model from disk")

  # evaluate loaded model on test data 
  # Define X_test & Y_test data first
  loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
  score = loaded_model.evaluate(mb.test_images, mb.test_labels, verbose=0)
  # Print accuracy etc  
  print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
except:
  # In the case a model wasn't found, make a new one!
  print("No model was found, creating new model ...")
  cm.createAndSaveModel()
  