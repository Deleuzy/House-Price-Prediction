#Here I have checked the lasso liniar regression method to see how this works
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the data
data = pd.read_csv("Housing.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data['mainroad'] = label_encoder.fit_transform(data['mainroad'])
data['guestroom'] = label_encoder.fit_transform(data['guestroom'])
data['basement'] = label_encoder.fit_transform(data['basement'])
data['hotwaterheating'] = label_encoder.fit_transform(data['hotwaterheating'])
data['airconditioning'] = label_encoder.fit_transform(data['airconditioning'])
data['prefarea'] = label_encoder.fit_transform(data['prefarea'])
data['furnishingstatus'] = label_encoder.fit_transform(data['furnishingstatus'])

# Split data into features and target variable
X = data.drop(columns=['price'])
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Min-Max scaler
scaler = MinMaxScaler()

# Normalize the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Lasso regression model
lasso_model = Lasso(alpha=1.0)  # You can adjust the alpha value to control the strength of regularization

# Train Lasso model on the training data
lasso_model.fit(X_train_scaled, y_train)

# Make predictions on the training set
lasso_train_pred = lasso_model.predict(X_train_scaled)

# Calculate training MSE and MAPE
lasso_train_mse = mean_squared_error(y_train, lasso_train_pred)
lasso_train_mape = mean_absolute_error(y_train, lasso_train_pred) / y_train.mean()

# Make predictions on the test set
lasso_test_pred = lasso_model.predict(X_test_scaled)

# Calculate testing MSE and MAPE
lasso_test_mse = mean_squared_error(y_test, lasso_test_pred)
lasso_test_mape = mean_absolute_error(y_test, lasso_test_pred) / y_test.mean()

print("Lasso Regression Model with Normalized Data:")
print("Training MSE:", lasso_train_mse)
print("Training MAPE:", lasso_train_mape)
print("Testing MSE:", lasso_test_mse)
print("Testing MAPE:", lasso_test_mape)
