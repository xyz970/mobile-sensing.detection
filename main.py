from flask import Flask, render_template, request,jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from PIL import Image
from tensorflow.keras.models import load_model
import os
import base64
import requests
import numpy as np
import io, base64
import cv2
import matplotlib.pyplot as plt
from predict import sequence_prediction

app = Flask(__name__)
app.debug = True


@app.route("/",methods=['POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        filename = secure_filename(file.filename)
        extension = filename.split('.')[1]
        newFilename = f"{str(uuid.uuid4())}.{extension}"
        folderPath = 'datasets/'+newFilename
        file.save(folderPath)
        return sequence_prediction(folderPath)



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
    app.run(host='192.168.0.115')
