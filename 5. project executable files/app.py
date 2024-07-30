from __future__ import division, print_function
import sys
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/inceptionV3-covid.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Rescale to [0, 1]

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        result = np.argmax(preds, axis=1)
        class_names = ['COVID', 'NORMAL', 'VIRAL_PNEUMONIA', 'LUNG_OPACITY']
        result = "The predicted output is "+class_names[result[0]]
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)




