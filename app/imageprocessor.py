# Helper libraries
import numpy as np
# Used for reading/helping process the image
from PIL import Image
from PIL import ImageOps as io
# Required for multi-dimensional image processing
from scipy import ndimage
# Required for image manipulation 
from cv2 import cv2
# Math is needed for fitting the image after downsizing it to 20x20
import math
# Local base file
import mnistbase as mb
# For deleting image after it's been processed
import os

'''
The main goal of this class and it's functions is 
to processes the canvas images in the same way 
that MNIST dataset images were processed.

Comments within the functions should provide an insight into how that function behaves.
Adapted from http://opensourc.es/blog/tensorflow-mnist
'''

# Adapted from https://stackoverflow.com/questions/41563720/error-when-checking-model-input-expected-convolution2d-input-1-to-have-4-dimens
# https://arrow.dit.ie/cgi/viewcontent.cgi?article=1190&context=scschcomdis - 3.8.1 Normalization and Reshape Data

# Takes in a black and white canvas image and processes it 
def preprocess_image(img):
    '''
    MNIST images are size normalized to fit in a 20x20 pixel box.
    They are then centered in a 28x28 image using the center of mass.
    This is a very important step in preprocessing if accuracy is to be maintained.
    '''

    # Crop out columns and rows that are completely black in the image
    while int(np.mean(img[0])) == 255:
        img = img[1:]

    while np.mean(img[:, 0]) == 255:
        img = np.delete(img, 0, 1)

    while np.mean(img[-1]) == 255:
        img = img[:-1]

    while np.mean(img[:, -1]) == 255:
        img = np.delete(img, -1, 1)

    # Resize it to the proper shape
    rows, cols = img.shape

    # Resize the outer box to fit it into a 20x20 box
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        img = cv2.resize(img, (cols, rows))

    # Still need a 28x28 image for the model, add the missing
    # rows and columns using np.lib.pad which adds to the sides.
    colsPadding = (int(math.ceil((28-cols)/2.0)),
                   int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),
                   int(math.floor((28-rows)/2.0)))
    img = np.lib.pad(img, (rowsPadding, colsPadding),
                     'constant', constant_values=255)

    # Need to now shift the image so that the digit's center of mass is centered appropriately
    shiftX, shiftY = get_best_shift(img)
    shifted = shift(img, shiftX, shiftY)
    img = shifted
    return img

# Calculate how to shift an image of a digit so that its center of mass is nicely centered.
def get_best_shift(img):
    # Calculate center of mass
    x, y = ndimage.measurements.center_of_mass(img)

    # Get the difference between the center of mass and image center to find shifts
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-x).astype(int)
    shifty = np.round(rows/2.0-y).astype(int)

    return shiftx, shifty

# Shifts an image by some offsets
def shift(img, sx, sy):
    # Generate warping transformation matrix to shift the image
    # https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html#gsc.tab=0
    warp = np.float32([[1, 0, sx], [0, 1, sy]])

    # Apply warping matrix
    rows, cols = img.shape
    shifted = cv2.warpAffine(img, warp, (cols, rows),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return shifted