# Here I train a linear regression model on the new_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("new_dataset.csv")

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns with mean
numerical_cols = data.select_dtypes(include=['number']).columns
imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# Perform one-hot encoding for categorical variables
data = pd.get_dummies(data, columns=categorical_cols)

# Split data into features and target variable
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_scaled, y_train)

# Make predictions
train_preds = linear_model.predict(X_train_scaled)
test_preds = linear_model.predict(X_test_scaled)

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
