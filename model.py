import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Data Preparation:
# URL of API endpoint
url = 'https://api.example.com/v1'

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

    # Learning Rate Scheduling:
    # Implement learning rate scheduling to adjust the learning rate during training to help the model converge faster and improve generalization
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True)
    
    # Update the optimizer to use the learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Model:
    # Build the neural network model
    # Model architecture with more hidden layers and increased number of neurons
    # Regularization techniques like dropout are applied to prevent overfitting
    # Applied random initialization by using different weight initialization techniques like He initialization to help prevent vanishing or exploding gradients
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)  # Output layer with one neuron for regression
    ])

    # Compile the model with the updated optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')

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
    
