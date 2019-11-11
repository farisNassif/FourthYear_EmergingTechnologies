![emergingtech](https://user-images.githubusercontent.com/22448079/47147656-9f228f80-d2c6-11e8-846a-aa6a9a88ffef.png)

<h3 align="center">A Neural Network for Predicting User-Drawn digits with Keras, OpenCV, Flask and MNIST </h3>

## Project Details

|Details  |    |
| --- | --- |
| **Link** | [Website](https://mnist-python-digit-prediction.herokuapp.com/)
| **Project**  | [Project Spec](https://github.com/ianmcloughlin/project-2019-emtech/blob/master/project.pdf) 
| **Course** | BSc (Hons) in Software Development
| **Module** |  Emerging Technologies |
| **College** | [GMIT](http://www.gmit.ie/) 
| **Author** | [Faris Nassif](https://github.com/farisNassif) |
| **Lecturer** | Dr Ian McLoughlin|

## Contents
* [Project Outline](#project-outline)
* [Running the Program](running-the-program)
* [Design](https://github.com) 
* [Software](#Software)
* [Resources](https://github.com)

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

1. In your command line terminal, `git clone https://github.com/farisNassif/FourthYear_EmergingTechnologies`
2. Navigate to the <b> \app\ </b> directory, `cd app`
3. Run the program from <b>runner.py</b>, `python runner.py`

Upon running you should be able to access the Web Application on `http://127.0.0.1:5000/`<br>
If you just want to try out the program without going through the chore of installing and running it you can [here](https://mnist-python-digit-prediction.herokuapp.com/)

<i> If for any reason after Step 3 you encounter errors after running, the most common issue would be missing imports, to solve this look for what import it says is missing and type `pip install whateverthatimportis`</i>



~Give a brief rundown of what is asked from the project<br>
~Outline of work eg. 'Create web app, users can draw on the page, submit the drawing and see if it's recognised'<br>
~How to run the project and it's requirements<br>
~Training the model<br>
~Accuracy of the model<br>
~Screenshots or a gif showing the project in action<br>
~Problems in the project<br>
~What I would change<br>
~Conclusion<br>

**TODO
clean up code majorly
have the drawn image saved or output or something
once thats done have it predicted
add comments 
add research notes
fix up canvas
make it work on any touch screen device
..

<p align="center">
  <img src = "https://i.imgur.com/TVplbBp.gif">
</p>
