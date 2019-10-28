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

import cv2 
from cv2 import cv2
import glob
import numpy as np
from PIL import Image


images = []
for filename in glob.glob('./TestImages/*.png'):
    im=Image.open(filename)
    images.append(im)

imgplot = plt.imshow(images[0])

plt.show()