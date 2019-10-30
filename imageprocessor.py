# Helper libraries
import numpy as np
# Used for reading/helping process the image
from PIL import Image
from PIL import ImageOps as io
import numpy as np
import matplotlib.pyplot as plt
# Local base file
import mnistbase as mb

# This function will take in a normal image file and reshape it so that it becomes compatible with the prediction model
def processImage(image):
    # Load the parameterized image
    loadedImg = Image.open(image)
    # Rezise it
    sized = io.fit(loadedImg, (28, 28)) 

    # Convert it to grayscale
    grey = sized.convert("L") 

    # Image needs to be black -or- white, use a lambda function to loop through each pixel of the image
    # If the pixel (x) is less than 128 set it to black (0,0,0), if greater set to white (255,255,255)
    im = grey.point((lambda x: 0 if x<128 else 255), '1') # Mode 1 (black/white)
    im.save('img.png') # Save the new image

    # Adapted from https://stackoverflow.com/questions/41563720/error-when-checking-model-input-expected-convolution2d-input-1-to-have-4-dimens
    imgArr = np.ndarray.flatten(np.array(im)).reshape(1, 784)
    # Reshape the image into the following dimensions
    imgArr = imgArr.reshape(imgArr.shape[0], 28, 28, 1)
    # Convert the float type
    imgArr = imgArr.astype('float32')
    # Returns the reshaped image
    return imgArr