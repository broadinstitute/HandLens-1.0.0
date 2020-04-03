# Handlens reader app

This is a mobile phone application that captures and analyzes an image of one or more lateral flow strips to quantify test results and resolve ambiguous readouts.

# What's in the repo?

This repo contains code for testing the strip detetection algorithms, the backend server that implements those algorithms as Python scripts, and the Android app frontend:

* android_app: Android Studio project for the Handlens Android app.
* notebooks: Jupyter notebooks to test the Computer Vision algorithm used to to detec the strips, extract the bands, and outout a prediction:
  - Strip Detection and Analysis.ipynb: It runs the entire image processing pipeline on one image at a time
  - Strip Prediction.ipynb: It runs the pipeline on a batch of images for which the ground truth is know, in order to compute the predictive performance of the algorithm (AUC, sensitivity, specificity, etc)
  - Threshold calculation.ipynb: It estimates the detection threshold that's needed for prediction
* scripts: the Python scripts that are runn on the server to process the images uploaded by the app and output a prediction back
* server: Node server which enables the Handlens app to upload and analyze images
