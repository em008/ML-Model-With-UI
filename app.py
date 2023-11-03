from flask import Flask, request, jsonify, render_template # Import libraries
import model_function  # Import ML model function module

# Initialize Flask app
app = Flask(__name__)

# Default home page
@app.route('/')
def home():
    return render_template('index.html')

# Route used for predictions in app UI   
@app.route('/predict', methods=['POST'])
def predict_count():
    features = [float(i) for i in request.form.values()]
    final_features = [np.array(features)]
    prediction = model_function.predict_count(final_features)

    result = round(prediction[0], 2)
    return render_template('index.html', prediction_result='CO2 Emission of the vehicle is :{}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
