import base64

from cv2 import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps as io

"""
This Class contains utility functions that can TODO
"""

def image_to_array(image):
    # Resize the image
    image = io.fit(image, (28,28))

    # Convert it for background purposes
    # Had a big issue with the alpha channel of the image, this was a workaround
    sized = image.convert('RGBA')

    # Create a 'background' that will be applied to the image
    # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil/9459208
    background = Image.new('RGBA', sized.size, (255,255,255))

    # Apply the background to the alpha channel of the RGBA image
    # When processed from the canvas, background is completely black, want a black number on a white background
    # TODO => Work without saving the image
    alpha_composite = Image.alpha_composite(background, sized)
    alpha_composite.save('toArray', 'PNG', quality=80)

    img = cv2.imread("toArray.png", cv2.IMREAD_GRAYSCALE)

    return img  