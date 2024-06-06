from flask import Flask, render_template, request,jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import os
import base64
import requests
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

