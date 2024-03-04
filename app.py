from flask import Flask, request, jsonify, render_template # Import libraries
import model  # Import ML model function module
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
    try:
        # Convert user input features to a NumPy array
        features = [int(i) for i in request.form.values()]
        final_features = [np.array(features)]

        # Use trained model and features to make prediction 
        trained_model = model.train_count_prediction_model(model.data)
        prediction = model.predict_count_for_date(trained_model, final_features)
    
        result = round(prediction, 2)
        return render_template('index.html', prediction_result='The scanned receipt count for this date is: {}'.format(result))

    except Exception as e:
        error_message = "An error occurred during prediction: {}".format(str(e))
        return render_template('index.html', prediction_result=error_message)
        
if __name__ == '__main__':
    app.run()
