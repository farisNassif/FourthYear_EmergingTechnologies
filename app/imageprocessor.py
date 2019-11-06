# Helper libraries
import numpy as np
# Used for reading/helping process the image
from PIL import Image
from PIL import ImageOps as io
# Local base file
import mnistbase as mb
# For deleting image after it's been processed
import os

"""
This Class functions as the processor for images that will be read by the model.
An image that is stored in memory is passed into processImage(), shaped and processed accordingly.

Comments within the functions should provide an insight into how that function behaves.
"""

# This function will take in a normal image file and reshape it so that it becomes compatible with the prediction model
def processImage(image):
    # Load the parameterized image
    loadedImg = image
    
    # Resize the image
    sized = io.fit(loadedImg, (28,28))

    # Convert it for background purposes
    # Had a big issue with the alpha channel of the image, this was a workaround
    sized = sized.convert('RGBA')

    # Create a 'background' that will be applied to the image
    # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil/9459208
    background = Image.new('RGBA', sized.size, (255,255,255))

    # Apply the background to the alpha channel of the RGBA image
    # When processed from the canvas, background is completely black, want a black number on a white background
    # TODO => Work without saving the image
    alpha_composite = Image.alpha_composite(background, sized)
    alpha_composite.save('toPredict.png', 'PNG', quality=80)

    # Save the processed image above
    predictme = Image.open('toPredict.png')

    # Convert it to grayscale
    predictme = predictme.convert("L") 
    
    # Image needs to be black -or- white, this lambda function loops through each pixel of the image
    # If the pixel (x) is less than 128 set it to black (0,0,0), if greater set to white (255,255,255)
    greyLambda = predictme.point((lambda x: 0 if x<128 else 255), '1') # Mode 1 (black/white)

    # Adapted from https://stackoverflow.com/questions/41563720/error-when-checking-model-input-expected-convolution2d-input-1-to-have-4-dimens
    imgArr = np.ndarray.flatten(np.array(greyLambda)).reshape(1, 784)
    # Reshape the image into the following dimensions
    imgArr = imgArr.reshape(imgArr.shape[0], 28, 28, 1)
    # Convert the float type
    imgArr = imgArr.astype('float32')

    # Remove saved image to clean up directory
    os.remove('toPredict.png')

    # Returns the reshaped image
    return imgArr
