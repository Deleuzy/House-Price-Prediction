# I have also tried to train an Neural network to see how well it can do here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset from CSV file
file_path = 'Housing.csv'  # Change the file name if needed
housing_data = pd.read_csv(file_path)

# Convert categorical variables to one-hot encoding
housing_data = pd.get_dummies(housing_data, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'])

# Define features and target variable
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=2500, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_scaled).flatten()  # Flatten the predicted values

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Calculate MAE
mae = np.mean(np.abs(y_test - y_pred))

print(f"Test MAPE: {mape:.2f}%")
print(f"Test MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
