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
    rm.predict("testpredict2.png")
    title = 'Draw a number!'
    # Base Page
    return render_template('layouts/index.html',
                       title=title)

# Upload/Publish route
@app.route('/upload', methods=['Post'])
def upload():
    # Requesting the drawn data through the AJAX function 
    image_b64=request.values[('imageBase64')]
    
    # Removing the first part of the base64 String, don't need it
    base64_data = re.sub('^data:image/.+;base64,', '', image_b64)
    print(base64_data)
    # Open the file that the base64 data will be written to
    output=open('output.png', 'wb')
    # Decode the data
    decoded=base64.b64decode(base64_data)
    # Write the decoded data to output.png, should now store the drawn image
    output.write(decoded)
    output.close()
    
    return "jeje"

def create_csv(text):
    unique_id = str(uuid.uuid4())
    with open('images/'+unique_id+'.png', 'a') as file:
        file.write(text[1:-1]+"\n")
    return unique_id

if __name__ == "__main__":
    # If theres any errors they'll pop up on the page
    app.run(debug=True)