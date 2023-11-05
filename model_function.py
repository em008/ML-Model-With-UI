import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Data Preparation:
# URL of the website or API endpoint to fetch data from
url = 'https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv'

# Load data
data = pd.read_csv(url)

# Ensure that the "Date" column is in a proper date format
data['Date'] = pd.to_datetime(data['# Date'])

def train_count_prediction_model(data):
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
    # Build the neural network model
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
    # Define early stopping
    # This stops training when a monitored metric has stopped improving, which can prevent overfitting and save training time
    # Training will stop if the validation loss (‘val_loss’) does not improve for 10 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    # Train the model using the training data
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    return model

def predict_count_for_date(trained_model, date_features):
    # Convert date_features to a NumPy array
    date_features = np.array(date_features).reshape(1, -1)

    # Make predictions using the trained model
    predicted_count = trained_model.predict(date_features)

    return predicted_count[0][0]

# Example:
if __name__ == "__main__":       
    # Train the model
    trained_model = train_count_prediction_model(data)

    # Predict the count for the given date
    date_features = [2022, 10]
    predicted_receipt_count = predict_count_for_date(trained_model, date_features)    
    print(f"Predicted receipt count for the given date: {predicted_receipt_count}")
    
