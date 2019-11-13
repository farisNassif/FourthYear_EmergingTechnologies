<p align="center">
  <img src = "https://i.imgur.com/fVVeh6U.png">
</p>

<h3 align="center">A Neural Network for Predicting User-Drawn digits with Keras, OpenCV, Flask and MNIST </h3>

## Project Details

|Details  |    |
| --- | --- |
| **Project**  | [Project Spec](https://github.com/ianmcloughlin/project-2019-emtech/blob/master/project.pdf) 
| **Link** | [Website](https://mnist-python-digit-prediction.herokuapp.com/)
| **Course** | BSc (Hons) in Software Development
| **Module** |  Emerging Technologies |
| **College** | [GMIT](http://www.gmit.ie/) 
| **Author** | [Faris Nassif](https://github.com/farisNassif) |
| **Lecturer** | Dr Ian McLoughlin|

## Contents
* [Project Outline](#project-outline)
* [Running the Program](#running-the-program)
* [Design and Technologies](#design-and-technologies) 
* [Resources](#resources)

## Project Outline
1. Create a Flask web application that allows users to draw digits on a Canvas.

2. The application should pre-process the drawn image before running it through a trained Model which should return a prediction for the drawn number back to the frontend user.

3. The development of the Model should be documented in a Jupyter Notebook, including clear explanations, documentation and training of the Model.

## Running the Program

### Requirements

In order to run the program on your machine, you must have the following installed

* [Python 3.5](https://www.python.org/downloads) (Or Above)
* [Tensorflow 2.0.0](https://www.tensorflow.org/install/pip)
* [OpenCV 2.4.5](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/) (Or Above)
* [Keras 2.3.0](https://keras.io/)

A handful of imports is also required, you can find those [here](https://github.com/farisNassif/FourthYear_EmergingTechnologies/blob/master/rough_work/required_imports.txt).

1. In your command line terminal: `git clone https://github.com/farisNassif/FourthYear_EmergingTechnologies`
2. Navigate to the <b> \app\ </b> directory: `cd app`
3. Run the program from <b>runner.py</b>: `python runner.py`

Upon running you should be able to access the Web Application on `http://127.0.0.1:5000/`.<br>
<i>If you just want to try out the program without going through the chore of installing and running it you can [here](https://mnist-python-digit-prediction.herokuapp.com/)</i>.

## Design and Technologies
### Neural Network
The [Neural Network Model](https://github.com/farisNassif/FourthYear_EmergingTechnologies/blob/master/model_notebook/ModelCreation.ipynb) is the crux of this project. The creation is made possible thanks to [Keras](https://keras.io/), a High-Level Neural Network API. The model itself isn't complicated thanks to Keras which encapsulates from the user a lot of the specificities and awkward parts of developing a Model.

The Model for this project has an accuracy of 99.16% making it highly accurate when tested against similar MNIST hand-written digits, however not too accurate when tested against unprocessed Canvas drawn digits, something I'll discuss when I break down the Backend. 

### Frontend
The web application is essentially a Single Page Application that communicates with the backend via [Flask](https://www.palletsprojects.com/p/flask/). 

The Frontend consists of a combination of HTML/Javascript/[Bootstrap](https://getbootstrap.com/) and [Jquery](https://jquery.com/). 

The user may draw a number (or anything really) on the canvas and 'Publish' their drawing. The drawing is sent to the backend via an [Ajax](https://api.jquery.com/jquery.ajax/) request. The drawing is transported as an [Encoded Base64](https://docs.python.org/2/library/base64.html) String which must be decoded, processed, compared against the trained Model which will return a prediction to the Frontend.

### Backend
Upon receiving the Base64 String it gets decoded into a .PNG image via the [Base64](https://docs.python.org/2/library/base64.html) Python module. The image is then stored in memory and processed with help from the [Pillow](https://pillow.readthedocs.io/en/stable/) module. The image needs to be pre-processed as an MNIST Dataset image would be processed otherwise the predictions will be largely inaccurate. I accomplished this with help from [OpenCV](https://opencv.org/). Images were converted into arrays and had to be reshaped into 20x20 images and then centered onto a 28x28 background, ensuring the digit's center of mass was centered appropriately.

Once processed the image is then compared against the previously trained Model, which returns a result. The result is then returned from Flask to the Frontend and displayed via the DOM to the user.

<p align="center">
  <img src = "https://i.imgur.com/MhecSHY.gif">
</p>
https://i.imgur.com/fVVeh6U.png
todo: conclusion, problems with project, add proper header img, proof read and fine tune before deadline
