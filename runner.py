from flask import Flask, render_template, request
import os
# Local file for running the model
import runmnist as rm
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
@app.route('/predict', methods=['Post'])
def upload():
    return "posted to backend!"

if __name__ == "__main__":
    # If theres any errors they'll pop up on the page
    app.run(debug=True)