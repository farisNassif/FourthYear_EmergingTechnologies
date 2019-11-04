from flask import Flask, render_template, request, jsonify
import os
import matplotlib.pyplot as plt
# Local file for running the model
import runmnist as rm
# Needed to decode the String data received from the canvas
import base64
# Regular expression matching, cuts off the unnecessary part of the Base64 String
import re
# Needed to store the image in memory rather than saving it on disk
import io 
# Used for opening the image in memory
from PIL import Image

app = Flask(__name__)
app._static_folder = os.path.abspath("templates/static/")

# Index route so when I browse to the url it doesn't 404
@app.route('/', methods=['Post', 'GET'])
def index():
    title = 'Draw a number!'
    # Base Page
    return render_template('layouts/index.html',
                       title=title)

# Upload/Publish route
# Saving the canvas data adapted from: https://www.science-emergence.com/Articles/How-to-convert-a-base64-image-in-png-format-using-python-/
@app.route('/upload', methods=['Post'])
def upload():
    # Requesting the drawn data through the AJAX function 
    image_b64=request.values[('imageBase64')]
    
    # Removing the first part of the base64 String, don't need it
    base64_data = re.sub('^data:image/.+;base64,', '', image_b64)

    # Decode the data
    decoded=base64.b64decode(base64_data)
    # In memory binary stream for the image received
    # https://docs.python.org/3/library/io.html
    inMemorySave = io.BytesIO(decoded)
    # Image in memory still needs to be opened before sent to be processed
    predictMe = Image.open(inMemorySave)
    # Pass the image and receive back a prediction
    res = rm.predict(predictMe)
    print("Prediction: " + str(res))
    return str(res)

if __name__ == "__main__":
    # If theres any errors they'll pop up on the page
    app.run(debug=True)