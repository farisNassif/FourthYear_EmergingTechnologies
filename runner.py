from flask import Flask, render_template, request, jsonify
import os
import matplotlib.pyplot as plt
# Local file for running the model
import runmnist as rm
import uuid
import base64
import re
# Having some issues with tensorflow and its depriciated packages

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
    print(base64_data)
    # Open the file that the base64 data will be written to
    output=open('DrawnNumber.png', 'wb')
    # Decode the data
    decoded=base64.b64decode(base64_data)
    # Write the decoded data to output.png, should now store the drawn image
    output.write(decoded)
    # Close the file
    output.close()
    
    res = rm.predict("DrawnNumber.png")
    print("Prediction: " + str(res))
    return str(res)

if __name__ == "__main__":
    # If theres any errors they'll pop up on the page
    app.run(debug=True)