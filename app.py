from flask import Flask, request, jsonify
import model_function  # Import ML model function module

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict_count():
    data = request.get_json()
    input_texts = [data['text1'], data['text2']]
    predictions = model_function.predict_count(input_texts)
    
    result = {'predition results': result}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
