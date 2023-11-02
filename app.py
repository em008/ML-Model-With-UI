from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Load model
model ='model.py'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = model.predict([text])

    result = {'category': prediction[0]}
    result = list(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
