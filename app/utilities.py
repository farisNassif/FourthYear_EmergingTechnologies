from cv2 import cv2
from PIL import Image
from PIL import ImageOps as io

"""
This Class contains utility functions that don't fit in any other class without causing clutter.

Comments within the functions should provide an insight into how that function behaves.
"""

# Takes in an image and processes it accordingly before transforming it into an array
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
    alpha_composite = Image.alpha_composite(background, sized)
    alpha_composite.save('toArray.png', 'PNG', quality=80)

    # Loads the image as greyscale
    image_array = cv2.imread("toArray.png", cv2.IMREAD_GRAYSCALE)

    return image_array  

# Takes an array as a paramater which should only contain values between 0 and 1
# The idea is to invert the canvas image as it would be inverted if it were to be an MNIST image
# https://stackoverflow.com/questions/19580102/inverting-image-in-python-with-opencv
def invert_values(image_array):

    # Need to make the array flat to allow looping
    array_flattened = image_array.flatten()

    # Inverts the values within the array { x => 1 - x } => E.G => { 0.4 = 1 - 0.4 } => x is now 0.6
    for i in range(array_flattened.size):
        array_flattened[i] = 1 - array_flattened[i]

    # Return the altered array with the same shape it had before
    return array_flattened.reshape(image_array.shape)