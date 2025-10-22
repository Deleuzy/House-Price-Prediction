
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # Add SimpleImputer import

# Load the data
data = pd.read_csv("housing_australia.csv")

# Split data into features and target variable
X = data.drop(columns=['price'])
y = data['price']

# One-hot encode categorical features
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Define the RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [75],     # Number of trees in the forest
    'max_depth': [8],             # Maximum depth of the trees
    'min_samples_leaf': [5],    # Minimum number of samples required to be at a leaf node
    'min_samples_split': [5]    # Minimum number of samples required to split an internal node
}

# Define Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=skf, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get the best estimator
best_rf_model = grid_search.best_estimator_

# Make predictions on training and testing data
train_preds = best_rf_model.predict(X_train_scaled)
test_preds = best_rf_model.predict(X_test_scaled)

# Calculate metrics
train_mse = mean_squared_error(y_train, train_preds)
train_mape = mean_absolute_error(y_train, train_preds) / y_train.mean() * 100
test_mse = mean_squared_error(y_test, test_preds)
test_mape = mean_absolute_error(y_test, test_preds) / y_test.mean() * 100

# Print results
print("Best Random Forest Model Parameters:")
print(grid_search.best_params_)
print("Training MSE:", train_mse)
print("Training MAPE:", train_mape)
print("Test MSE:", test_mse)
print("Test MAPE:", test_mape)
