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
    
    # Resize the image
    sized = io.fit(loadedImg, (28,28))
    sized.save('sized.png')

    sized = Image.open('sized.png').convert('RGBA')
    background = Image.new('RGBA', sized.size, (255,255,255))

    alpha_composite = Image.alpha_composite(background, sized)
    alpha_composite.save('toPredict.png', 'PNG', quality=80)

    predictme = Image.open('toPredict.png')

    # Convert it to grayscale
    predictme = predictme.convert("L") 
    predictme.save('PLEASE.PNG')
    
    # Image needs to be black -or- white, use a lambda function to loop through each pixel of the image
    # If the pixel (x) is less than 128 set it to black (0,0,0), if greater set to white (255,255,255)
    AfterPoint = predictme.point((lambda x: 0 if x<128 else 255), '1') # Mode 1 (black/white)
    AfterPoint.save('AfterPoint.png') 

    # Adapted from https://stackoverflow.com/questions/41563720/error-when-checking-model-input-expected-convolution2d-input-1-to-have-4-dimens
    imgArr = np.ndarray.flatten(np.array(AfterPoint)).reshape(1, 784)
    # Reshape the image into the following dimensions
    imgArr = imgArr.reshape(imgArr.shape[0], 28, 28, 1)
    # Convert the float type
    imgArr = imgArr.astype('float32')
    # Returns the reshaped image
    return imgArr