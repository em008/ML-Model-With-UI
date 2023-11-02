import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Data Preparation:
# URL of the website or API endpoint to fetch data from
url = 'https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv'

# Load data
data = pd.read_csv(url)

# Ensure that the "Date" column is in a proper date format
data['Date'] = pd.to_datetime(data['# Date'])

# print(data)

# Feature Engineering:
# Extract various features like year, month, day from date data
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Split Data:
# Define features and target variable
X = data[['Year', 'Month']]
y = data['Receipt_Count']

# Split the data into training and testing sets
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Choose a Model:
# Build a neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Output layer with one neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Convert data to NumPy arrays
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

# Training:
# Train model using the training data
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Prediction and Evaluation:
# Use the trained model to predict receipt counts for the test set and evaluate its performance
mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")

# Make Predictions:
# Once the model is trained and evaluated make predictions on future dates by providing the corresponding feature values
future_date_features = np.array([[2022, 11]])
predicted_receipt_count = model.predict(future_date_features)
print(f"Predicted Receipt Count for Future Date: {predicted_receipt_count[0][0]}")
