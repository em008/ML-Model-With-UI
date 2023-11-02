from flask import Flask, request, jsonify
import model  # Import ML model module

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Load model
ml_model = model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = ml_model.predict([text])

    result = {'category': prediction[0]}
    result = list(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
