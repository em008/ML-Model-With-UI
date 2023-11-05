from flask import Flask, request, jsonify, render_template # Import libraries
import model_function  # Import ML model function module
import pandas as pd
import numpy as np 

# Initialize Flask app
app = Flask(__name__)

# Default home page
@app.route('/')
def home():
    return render_template('index.html')

# Route used for predictions in app UI   
@app.route('/predict', methods=['POST'])
def predict():
    features = [int(i) for i in request.form.values()]
    final_features = [np.array(features)]

    trained_model = model_function.train_count_prediction_model(model_function.data)
    prediction = model_function.predict_count_for_date(trained_model, final_features)

    result = round(prediction, 2)
    return render_template('index.html', prediction_result='The scanned receipt count for this date is :{}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
