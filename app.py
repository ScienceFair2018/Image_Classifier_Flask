from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)


MODEL_PATH = 'wounds_model.h5'
# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()

# #For creating model
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('model_resnet.h5')

import keras
def model_predict(img_path, model):

    categories = ['Background', 'Diabetic Wound', 'Normal', 'Pressure Wound', 'Surgical Wound', 'Venous Wound']

    # preprocessing
    img = keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.utils.normalize(x)

    # make prediction
    preds = model.predict(x)
    print (categories)
    print (preds)
    category = categories[np.argmax(preds[0])]
    if (category == 'Background' or 'Surgical Wound' or 'Normal') and (preds[0][np.argmax(preds)] * 100 < 80):
        if (preds[0][1] > preds[0][5]):
            category = 'Diabetic Wound'
        if (preds[0][5] > preds[0][3]):
            category = 'Venous Wound'
        else:
            category = 'Pressure Wound'
    return ("Model predicts a " + category)


@app.route('/Image_Classification', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'test', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print (preds)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)
        #pred_class = decode_predictions(preds, top=1)
        #result = str(pred_class[0][0][1])
        return str(preds)
    return None

if __name__ == '__main__':
      app.run()
