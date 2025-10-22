# Here I also tried to do a CNN to see how well it can do but it did not touch the winnig model Cat_boost

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
import numpy as np
from sklearn.metrics import mean_squared_error

Times=2500

# Load the dataset from CSV file in Google Colab
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

# Reshape data for CNN input (assuming each row as an image)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Build the CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=Times , batch_size=8, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_reshaped).flatten()  # Flatten the predicted values
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)

print(f"Test MAPE: {mape:.2f}%")
print(f"Test MSE: {rmse:.2f}")
print(f"Test MAE: {mse:.2f}")
