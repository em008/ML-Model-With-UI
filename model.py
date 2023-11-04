from tensorflow import keras
import pandas as pd

def prediction_model(data):
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

    return model

# Example usage:
if __name__ == "__main__":
    # Data Preparation:
    # URL of the website or API endpoint to fetch data from
    url = 'https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv'

    # Load data
    data = pd.read_csv(url)

    # Ensure that the "Date" column is in a proper date format
    data['Date'] = pd.to_datetime(data['# Date'])

    # Create and train the model
    trained_model = prediction_model(data)
    
    # Example: Make predictions for future dates
    future_date_features = pd.DataFrame({'Year': [2022], 'Month': [10]})
    predicted_receipt_count = trained_model.predict(future_date_features)
    print(f"Predicted Receipt Count for Future Date: {predicted_receipt_count[0][0]}")
