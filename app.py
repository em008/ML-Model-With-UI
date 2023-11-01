from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name)

# Load model
model ='model.py'

@app.route('/predict', methods=['POST'])
def predict():
    prediction = model.predict()

    result = {if prediction[0][0] else 'error': 'Try again'}
    return jsonify(result)

if __name__ == '__main__':
    app.run()
