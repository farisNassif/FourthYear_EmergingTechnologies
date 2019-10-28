# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# Used for predicting local image
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageOps as io

def processImage(image):
    loadedImg = Image.open(image)
    sized = io.fit(loadedImg, (28, 28)) # Rezise it
    grey = sized.convert("L") # Convert it to grayscale
    plt.imshow(grey)
    plt.show()

    # Adapted from https://stackoverflow.com/questions/41563720/error-when-checking-model-input-expected-convolution2d-input-1-to-have-4-dimens
    imgArr = np.ndarray.flatten(np.array(grey)).reshape(1, 784)
    imgArr = imgArr.reshape(imgArr.shape[0], 28, 28, 1)
    imgArr = imgArr.astype('float32')

    return imgArr