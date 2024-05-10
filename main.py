from flask import Flask, render_template, request
import os
import base64
import requests
import numpy as np
import io, base64
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.debug = True


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict/image', methods=['POST'])
def predict():
    classes = [
        "0", "1"
    ]
    file = request.files['image']
    img = Image.open(file.stream)
    
    # data = request.data
    # data = base64.b64decode(data)
    # print(data)
    # img = Image.open(io.BytesIO(data))
    img.save('static/img/predict.png')
    model = load_model('Model_cnn.h5')
    imageArray = cv2.imread('static/img/predict.png')
    newImageArray = cv2.resize(imageArray, (100, 100))
    image = np.array(newImageArray, dtype="float32")
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    index = np.argmax(prediction[0], axis=0)
    print('prediction:', index)
    return classes[index]


if __name__ == '__main__':
    app.run()
