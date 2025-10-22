# In this step I have made an initial trial to train a DNN model on the Housing.csv dataset and see how well does it do in 100 epochs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
data = pd.read_csv("new_dataset.csv")

# Separate categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=['number']).columns

# Handle missing values for categorical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

# Handle missing values for numerical columns
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])

# Split data into features and target variable
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# One-hot encode categorical features
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # No activation function for the output layer (linear activation)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=1000, batch_size=8, verbose=1)

# Make predictions
train_preds = model.predict(X_train_scaled).flatten()
test_preds = model.predict(X_test_scaled).flatten()

# Calculate metrics
train_mse = mean_squared_error(y_train, train_preds)
train_mape = mean_absolute_error(y_train, train_preds) / y_train.mean() * 100
test_mse = mean_squared_error(y_test, test_preds)
test_mape = mean_absolute_error(y_test, test_preds) / y_test.mean() * 100

# Print results
print("Training MSE:", train_mse)
print("Training MAPE:", train_mape)
print("Test MSE:", test_mse)
print("Test MAPE:", test_mape)
