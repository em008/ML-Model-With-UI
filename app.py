from flask import Flask, request, jsonify, render_template # Import libraries
import model_function  # Import ML model function module
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Default home page
@app.route('/')
def home():
    return render_template('index.html')

# Route used for predictions in app UI   
@app.route('/predict', methods=['POST'])
def predict():
    # features = [int(i) for i in request.form.values()]
    # final_features = [np.array(features)]
    # prediction = model_function.prediction_model(final_features)

    form_data = request.form.to_dict()
    features = pd.DataFrame(form_data, index=[0])
    prediction = model_function.prediction_model.predict(features)

    result = round(prediction[0], 2)
    return render_template('index.html', prediction_result='The scanned receipt count for this date is :{}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
