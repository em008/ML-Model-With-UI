from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO

app = Flask(__name)

# Load a pre-trained model for image classification
model = keras.models.load_model('cat_dog_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.asarray(image)
    image = (image - 127.5) / 127.5  # Normalize the image
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = Image.open(file)
    image = preprocess_image(image)
    prediction = model.predict(np.array([image]))

    result = {'class': 'cat' if prediction[0][0] > 0.5 else 'dog'}
    return jsonify(result)

if __name__ == '__main__':
    app.run()
