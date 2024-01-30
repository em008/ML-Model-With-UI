# ML-Model-With-UI

For this project a neural network regression model is built using TensorFlow and Keras to predict future counts based on data collected. The model architecture has several layers and number of neurons. Regularization techniques like dropout are applied to prevent overfitting. A simple web application is built using Flask that integrates the model. 

A neural network regression model was selected because the architecture can make predictions for continuous values. It leverages hidden layers, activation functions, optimization algorithms, and regularization techniques to learn complex patterns in the data and produce accurate continuous predictions. Furthermore, neural network regression models have hyperparameters, such as the number of hidden layers, the number of nodes in each layer, learning rate, and regularization strength. Hyperparameter tuning involves finding the optimal set of hyperparameters for making predictions.

<!---
To access the app in an internet browser use this [link](https://em008.github.io/ML-Model-With-UI/).
-->

To run the app locally open a terminal or CLI and navigate to the directory where the `requirements.txt` file is located. Next, run the following command to install the packages listed in `requirements.txt`: `pip install -r requirements.txt`. Then, navigate to the directory where the `app.py` file is located and run the command: `python app.py`. The app will run on this local development [server](http://127.0.0.1:5000) by deafult unless manually changed.

Tech Stack
- Machine Learning Model: Python, Tensorflow, Keras
- Backend: Python, Flask
- Frontend: HTML, CSS
